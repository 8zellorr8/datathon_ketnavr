"""Portable local launcher. No video analysis, training or public publishing here."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "data" / "runtime"
PORTABLE = RUNTIME / "portable"
MODEL = "qwen3-vl:2b-instruct-q4_K_M"
MIN_PYTHON = (3, 10)


class LaunchError(Exception):
    pass


def say(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + secrets.token_hex(4))
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LaunchError("Перенаправление локального сервиса запрещено.")


def request_json(origin: str, path: str, payload=None, timeout: float = 3) -> dict:
    # Origins are constructed exclusively from numeric loopback ports below.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(origin + path, data=data,
                                     headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise LaunchError("Слишком большой ответ локального сервиса.")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise LaunchError("Некорректный ответ локального сервиса.")
    return result


def get_json(origin: str, path: str) -> dict | None:
    try:
        return request_json(origin, path)
    except (OSError, ValueError, LaunchError):
        return None


def port_is_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        return connection.connect_ex(("127.0.0.1", port)) == 0


def find_ollama() -> str | None:
    installed = shutil.which("ollama")
    if installed:
        return installed
    candidates = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe",
                  Path("/Applications/Ollama.app/Contents/Resources/ollama")]
    return next((str(path) for path in candidates if path.is_file()), None)


def compatible_python(executable: Path) -> bool:
    if not executable.is_file():
        return False
    code = "import sys,struct;raise SystemExit(0 if sys.version_info>=(3,10) and struct.calcsize('P')==8 else 1)"
    try:
        result = subprocess.run([str(executable), "-c", code], cwd=ROOT,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def venv_python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def prepare_python(offline: bool) -> Path:
    # A copied environment may be unusable; never delete or reset it.
    for name in (".venv-portable", ".venv"):
        executable = venv_python(ROOT / name)
        if compatible_python(executable):
            chosen = executable
            break
    else:
        if offline:
            raise LaunchError("Нет готового Python-окружения. Первый запуск выполните без --offline.")
        destination = ROOT / ".venv-portable"
        if destination.exists():
            destination = ROOT / (".venv-portable-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
        say("Создаю отдельное окружение Python…")
        result = subprocess.run([sys.executable, "-m", "venv", str(destination)], cwd=ROOT)
        if result.returncode:
            raise LaunchError("Не удалось создать venv. Для Linux может потребоваться пакет python3-venv.")
        chosen = venv_python(destination)
    check = (
        "import importlib.metadata,re;"
        "version=lambda name:tuple(map(int,re.findall(r'[0-9]+',importlib.metadata.version(name))[:3]));"
        "ok=(0,6)<=version('imageio-ffmpeg')<(0,7) and (10,4)<=version('pillow')<(13,);"
        "raise SystemExit(0 if ok else 1)"
    )
    result = subprocess.run([str(chosen), "-c", check], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode:
        if offline:
            raise LaunchError("Зависимости отсутствуют или отличаются. Первый запуск нужен без --offline.")
        say("Устанавливаю зависимости из requirements.txt…")
        result = subprocess.run([str(chosen), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], cwd=ROOT)
        if result.returncode:
            raise LaunchError("Установка зависимостей не завершилась. Проверьте интернет и место на диске.")
    ffmpeg_check = (
        "from forte_ai.video import ffmpeg_executable;import subprocess;"
        "subprocess.run([ffmpeg_executable(),'-version'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
    )
    result = subprocess.run([str(chosen), "-c", ffmpeg_check], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    if result.returncode:
        raise LaunchError("FFmpeg не работает на этой ОС/архитектуре. Установите FFmpeg для своей ОС и добавьте его в PATH либо задайте FORTE_FFMPEG.")
    return chosen


def ollama_environment(profile: dict, port: int, force_cpu: bool) -> dict:
    environment = os.environ.copy()
    environment.update({
        "OLLAMA_HOST": f"127.0.0.1:{port}", "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_CONTEXT_LENGTH": str(profile["num_ctx"]),
        "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0",
        "LLAMA_ARG_CACHE_RAM": str(profile["cache_ram_mib"]),
    })
    if force_cpu:
        environment.update({"CUDA_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1",
                            "GGML_VK_VISIBLE_DEVICES": "-1", "OLLAMA_VULKAN": "0"})
    return environment


def start_child(command: list[str], environment: dict, log_name: str, owned: list) -> subprocess.Popen:
    log = (PORTABLE / log_name).open("ab")
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                   start_new_session=(os.name != "nt"))
    finally:
        log.close()
    owned.append(process)
    return process


def stop_owned(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        # The PID is from our Popen, never an existing user's Ollama/API process.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)


def wait_service(origin: str, path: str, predicate, process: subprocess.Popen, seconds=45) -> dict:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        response = get_json(origin, path)
        if response is not None and predicate(response):
            return response
        if process.poll() is not None:
            break
        time.sleep(0.5)
    raise LaunchError("Сервис не запустился. Технические логи: data/runtime/portable/.")


def ensure_model(ollama: str, origin: str, environment: dict, config: dict, offline: bool) -> dict:
    tags = request_json(origin, "/api/tags")
    installed = next((item for item in tags.get("models", [])
                      if item.get("name") == MODEL or item.get("model") == MODEL), None)
    if installed is None:
        if offline:
            raise LaunchError("Локальные веса Qwen не найдены. Первый запуск нужен без --offline.")
        say(f"Скачиваю только веса {MODEL}. Видео при этом не передаются…")
        result = subprocess.run([ollama, "pull", MODEL], cwd=ROOT, env=environment)
        if result.returncode:
            raise LaunchError("Не удалось скачать веса. Проверьте сеть, свободное место и версию Ollama.")
        tags = request_json(origin, "/api/tags")
        installed = next((item for item in tags.get("models", [])
                          if item.get("name") == MODEL or item.get("model") == MODEL), None)
    if not installed:
        raise LaunchError("После скачивания локальная модель не найдена.")
    detail = request_json(origin, "/api/show", {"model": MODEL}, timeout=10)
    for item in (installed, detail, detail.get("details", {})):
        if item.get("remote_host") or item.get("remote_model"):
            raise LaunchError("Облачные модели запрещены: требуется локальная Qwen.")
    if "vision" not in detail.get("capabilities", []):
        raise LaunchError("Установленная модель не подтверждает поддержку изображений.")
    expected = config.get("expected_digest")
    if expected and installed.get("digest") != expected:
        raise LaunchError("Веса отличаются от зафиксированной версии. Нужна отдельная проверка версии модели; автоматическая подмена отключена.")
    return {"name": MODEL, "digest": installed.get("digest"), "vision_verified": True}


def api_key(environment: dict) -> Path:
    key_path = RUNTIME / "api-key.txt"
    if environment.get("FORTE_API_KEY"):
        if len(environment["FORTE_API_KEY"]) < 16:
            raise LaunchError("FORTE_API_KEY должен содержать не менее 16 символов.")
        say("API использует ключ из FORTE_API_KEY.")
        return key_path
    if not key_path.exists():
        with key_path.open("x", encoding="ascii") as stream:
            stream.write(secrets.token_urlsafe(32))
        try:
            key_path.chmod(0o600)
        except OSError:
            pass
    key = key_path.read_text(encoding="ascii").strip()
    if len(key) < 16:
        raise LaunchError("Локальный файл ключа некорректен; существующий ключ не заменён.")
    environment["FORTE_API_KEY"] = key
    return key_path


def describe(hardware: dict, profile: dict) -> None:
    say(f"ОС: {hardware.get('platform')}; Python: {hardware.get('python_bits')}-бит.")
    memory = hardware.get("ram", {})
    say(f"RAM: {memory.get('total_mib')} МиБ; свободно: {memory.get('available_mib')} МиБ.")
    gpus = hardware.get("gpus", [])
    for gpu in gpus:
        say(f"GPU: {gpu.get('name')}; память: {gpu.get('total_mib')} МиБ ({gpu.get('source')}).")
    if not gpus:
        say("Поддерживаемая GPU не подтверждена. Ollama попробует доступный backend/процессор.")
    say(f"Профиль: {profile['name']}; кеш RAM: {profile['cache_ram_mib']} МиБ.")
    for message in profile.get("warnings", []) + profile.get("recommendations", []):
        say(message)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description="Автоматический локальный запуск ForteBank")
    parser.add_argument("--check", action="store_true", help="Только диагностика; без установок и запуска")
    parser.add_argument("--startup-check", action="store_true", help="Поднять сервисы, проверить готовность без видео и завершить собственные процессы")
    parser.add_argument("--offline", action="store_true", help="Запретить установку зависимостей и скачивание весов")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Принудительный CPU-профиль; отдельный порт Ollama")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model-port", type=int, default=11435)
    args = parser.parse_args()
    if sys.version_info < MIN_PYTHON or struct.calcsize("P") != 8:
        say("Нужен 64-битный Python 3.10+, рекомендуется 3.12: https://www.python.org/downloads/")
        return 2
    if not all(1024 <= port <= 65535 for port in (args.port, args.model_port)) or args.port == args.model_port:
        say("Нужны разные локальные порты от 1024 до 65535.")
        return 2
    if args.cpu and sys.platform == "darwin":
        say("Принудительный --cpu на macOS пока не подтверждён. Запустите без этого флага: Ollama сама выберет Metal/CPU.")
        return 2
    from portable_hardware import detect_hardware, select_profile
    hardware = detect_hardware()
    profile = select_profile(hardware)
    if args.cpu:
        profile = {**profile, "name": "cpu_forced", "backend_hint": "cpu",
                   "cache_ram_mib": 128}
    describe(hardware, profile)
    ollama = find_ollama()
    if args.check:
        say(json.dumps({"hardware": hardware, "profile": profile,
                        "ollama_installed": bool(ollama), "model": MODEL}, ensure_ascii=False))
        return 0
    url = f"http://127.0.0.1:{args.port}"
    if port_is_busy(args.port):
        health = get_json(url, "/health")
        if (health and health.get("status") == "ok" and health.get("backend") == "local"
                and health.get("ui") == "fortelens" and not args.cpu and args.model_port == 11435):
            say("Сайт уже запущен: " + url)
            say("Повторный экземпляр не создаётся. Его модель, ключ и параметры памяти этим запуском не менялись.")
            if not args.no_browser:
                webbrowser.open(url)
            return 0
        say("Порт сайта занят другим сервисом или старым интерфейсом. Для отдельного запуска задайте свободный --port, например --port 8003.")
        return 2
    if not ollama:
        say("Установите Ollama для своей ОС: https://ollama.com/download. Затем запустите этот файл снова.")
        return 2
    PORTABLE.mkdir(parents=True, exist_ok=True)
    owned: list[subprocess.Popen] = []
    try:
        config = json.loads((ROOT / "config/local.json").read_text(encoding="utf-8"))
        if config.get("model") != MODEL:
            raise LaunchError("Модель в конфигурации отличается от проверенного профиля.")
        python = prepare_python(args.offline)
        environment = ollama_environment(profile, args.model_port, args.cpu)
        origin = f"http://127.0.0.1:{args.model_port}"
        version = get_json(origin, "/api/version")
        model_reused = bool(version and version.get("version"))
        if version and version.get("version"):
            if args.cpu:
                raise LaunchError("Для --cpu задайте свободный --model-port: параметры уже работающей Ollama не меняются.")
            say("Использую уже работающую локальную Ollama; её параметры памяти остаются прежними.")
        else:
            if port_is_busy(args.model_port):
                raise LaunchError("Порт модели занят другим сервисом. Укажите свободный --model-port.")
            say("Запускаю Ollama только на 127.0.0.1…")
            model_process = start_child([ollama, "serve"], environment, "ollama.log", owned)
            version = wait_service(origin, "/api/version", lambda item: bool(item.get("version")), model_process)
        model = ensure_model(ollama, origin, environment, config, args.offline)
        config.update({"base_url": origin, "rules_path": str(ROOT / "config/rules.json"),
                       "frontend_dir": str(ROOT.parent / "frontend/dist"),
                       "visual_checks_path": str(ROOT / "config" / config["visual_checks_path"])
                       if config.get("visual_checks_path") else None,
                       "runtime_dir": str(RUNTIME), "num_ctx": profile["num_ctx"],
                       "request_timeout_sec": profile["request_timeout_sec"]})
        config_path = PORTABLE / "config.auto.json"
        write_json(config_path, config)
        from portable_data import rebase_manifest
        data_report = rebase_manifest(ROOT, RUNTIME)
        say("Данные: " + data_report.get("status", "unknown") + ". Разделение выборок сохранено.")
        if data_report.get("status") == "rebased" and data_report.get("training_warning"):
            say(data_report["training_warning"])
        if port_is_busy(args.port):
            raise LaunchError("Порт сайта успел занять другой процесс. Укажите свободный --port.")
        api_key(environment)
        api_process = start_child([str(python), "-m", "forte_ai", "serve", "--config", str(config_path),
                                   "--host", "0.0.0.0", "--port", str(args.port)], environment, "api.log", owned)
        wait_service(url, "/health", lambda item: item.get("status") == "ok" and item.get("backend") == "local"
                     and item.get("ui") == "fortelens", api_process)
        write_json(PORTABLE / "launch.json", {
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "hardware": hardware,
            "selected_profile": profile, "ollama": version, "model": model,
            "api_origin": url, "model_origin": origin, "data": data_report,
            "model_weights_fine_tuned": False, "video_inference_performed_by_launcher": False,
            "model_already_running": model_reused,
            "ollama_environment_applied": not model_reused,
        })
        say("Сайт: " + url)
        if environment.get("FORTE_API_KEY") and os.environ.get("FORTE_API_KEY"):
            say("Вставьте в форму свой ключ FORTE_API_KEY.")
        else:
            say("Ключ для формы находится в файле: " + str(RUNTIME / "api-key.txt"))
        say("GPU обнаружена по данным системы; реальная скорость и размещение модели ещё не измерены этим запускателем.")
        say("Автозапуск не дообучает модель и не исправляет качество распознавания.")
        if args.startup_check:
            say("Локальный запуск подтверждён. Анализ видео не выполнялся; завершаю собственные процессы.")
            return 0
        if not args.no_browser:
            webbrowser.open(url)
        say("Оставьте это окно открытым. Ctrl+C завершит только сервисы, запущенные этим окном.")
        while all(process.poll() is None for process in owned):
            time.sleep(0.5)
        raise LaunchError("Один из запущенных сервисов завершился. См. data/runtime/portable/.")
    except KeyboardInterrupt:
        say("Останавливаю запущенные этим окном сервисы…")
        return 0
    except (LaunchError, OSError, ValueError, subprocess.SubprocessError) as error:
        say("Запуск не завершён: " + str(error))
        return 2
    finally:
        for process in reversed(owned):
            try:
                stop_owned(process)
            except (OSError, subprocess.SubprocessError):
                say("Не удалось полностью остановить один из собственных дочерних процессов.")


if __name__ == "__main__":
    raise SystemExit(main())
