"""Event verification metadata checks, using synthetic JPEG markers only."""

import base64
import copy
import unittest

from forte_ai.server import _validate_analysis_result


def legacy_event():
    return {
        "type": "Synthetic observation",
        "description": "Synthetic action for result validation only.",
        "timecode_sec": 6.0,
        "evidence_frames": [{
            "timestamp_sec": 6.0,
            "image_base64": base64.b64encode(b"\xff\xd8synthetic\xff\xd9").decode("ascii"),
        }],
    }


def observation_event():
    return {
        **legacy_event(),
        "verification_status": "needs_regulation_review",
        "rule_source": None,
        "detection_basis": "visual_observation",
        "requires_human_review": True,
        "regulatory_note": "The relevant regulation still needs review.",
    }


def verified_source_event():
    return {
        **legacy_event(),
        "verification_status": "normative_source_verified",
        "rule_source": {"verified": True, "title": "Synthetic source"},
        "detection_basis": "verified_rule",
        "requires_human_review": True,
    }


class EventVerificationTests(unittest.TestCase):
    def validate(self, *events):
        return _validate_analysis_result({
            "status": "done",
            "verdict": "violation" if events else "no_violation",
            "violations": list(events),
        })

    def test_legacy_event_without_metadata_remains_supported(self):
        self.validate(legacy_event())

    def test_visual_observation_with_unconfirmed_source_is_supported(self):
        self.validate(observation_event())

    def test_verified_source_event_still_requires_human_review(self):
        self.validate(verified_source_event())

    def test_mixed_legacy_verified_and_observation_result_is_supported(self):
        self.validate(legacy_event(), verified_source_event(), observation_event())

    def test_empty_completed_result_is_supported(self):
        self.validate()

    def test_present_status_must_be_an_allowed_string(self):
        for status in (None, False, 1, "", "verified", [], {}):
            with self.subTest(status=status):
                event = observation_event()
                event["verification_status"] = status
                with self.assertRaises(ValueError):
                    self.validate(event)

    def test_observation_rejects_any_citation_including_verified_source(self):
        for source in ({"verified": True}, {"verified": False}, {}, "source", []):
            with self.subTest(source=source):
                event = observation_event()
                event["rule_source"] = source
                with self.assertRaises(ValueError):
                    self.validate(event)

    def test_observation_requires_explicit_null_source(self):
        event = observation_event()
        del event["rule_source"]
        with self.assertRaises(ValueError):
            self.validate(event)

    def test_verified_source_requires_a_dict_and_exact_true(self):
        for source in (None, [], "source", {}, {"verified": False}, {"verified": 1}, {"verified": None}):
            with self.subTest(source=source):
                event = verified_source_event()
                event["rule_source"] = source
                with self.assertRaises(ValueError):
                    self.validate(event)

    def test_both_statuses_require_the_matching_detection_basis(self):
        for factory in (observation_event, verified_source_event):
            valid = factory()
            incorrect = "verified_rule" if valid["detection_basis"] == "visual_observation" else "visual_observation"
            for basis in (incorrect, None, False, ""):
                with self.subTest(status=valid["verification_status"], basis=basis):
                    event = copy.deepcopy(valid)
                    event["detection_basis"] = basis
                    with self.assertRaises(ValueError):
                        self.validate(event)
            event = copy.deepcopy(valid)
            del event["detection_basis"]
            with self.assertRaises(ValueError):
                self.validate(event)

    def test_both_statuses_require_exact_true_human_review_flag(self):
        for factory in (observation_event, verified_source_event):
            for review in (False, None, 1, "true"):
                with self.subTest(status=factory()["verification_status"], review=review):
                    event = factory()
                    event["requires_human_review"] = review
                    with self.assertRaises(ValueError):
                        self.validate(event)
            event = factory()
            del event["requires_human_review"]
            with self.assertRaises(ValueError):
                self.validate(event)

    def test_optional_regulatory_note_must_be_a_string(self):
        for note in (None, False, 1, [], {}):
            with self.subTest(note=note):
                event = observation_event()
                event["regulatory_note"] = note
                with self.assertRaises(ValueError):
                    self.validate(event)


if __name__ == "__main__":
    unittest.main()
