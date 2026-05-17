from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datacenter_verification.observable_algorithm import evaluate_site


class ObservableAlgorithmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = json.loads((ROOT / "synthetic" / "sites.json").read_text(encoding="utf-8"))
        cls.sites = {site["scenario_key"]: site for site in payload["sites"]}
        cls.results = {key: evaluate_site(site) for key, site in cls.sites.items()}

    def result(self, key: str) -> dict:
        return self.results[key]

    def stage(self, key: str, stage: str) -> dict:
        return self.result(key)["stage_outputs"][stage]

    def test_all_synthetic_expected_outputs_match(self) -> None:
        for key, site in self.sites.items():
            with self.subTest(key=key):
                result = self.results[key]
                expected = site["expected"]
                self.assertEqual(expected["A_capacity_gate_label"], self.stage(key, "A_capacity_gate")["label"])
                self.assertEqual(expected["final_route"], result["final_route"])
                self.assertEqual(expected["capacity_short_circuit"], self.stage(key, "A_capacity_gate")["short_circuited"])
                b_labels = self.stage(key, "B_training_candidate_detection")["labels"]
                c_labels = self.stage(key, "C_discrepancy_and_explanation_review")["labels"]
                for label in expected["B_training_candidate_detection_labels"]:
                    self.assertIn(label, b_labels)
                for label in expected["C_discrepancy_and_explanation_review_labels"]:
                    self.assertIn(label, c_labels)

    def test_capacity_ruleout_short_circuits_b_and_c(self) -> None:
        result = self.result("C_capacity_ruled_out")
        self.assertEqual("capacity_ruled_out_for_scope", result["final_route"])
        self.assertTrue(self.stage("C_capacity_ruled_out", "A_capacity_gate")["short_circuited"])
        self.assertEqual("skipped_due_to_capacity_ruleout", self.stage("C_capacity_ruled_out", "B_training_candidate_detection")["mode"])
        self.assertEqual("skipped_due_to_capacity_ruleout", self.stage("C_capacity_ruled_out", "C_discrepancy_and_explanation_review")["mode"])

    def test_clean_training_reaches_high_warning(self) -> None:
        result = self.result("A_clean_threshold_training")
        b = self.stage("A_clean_threshold_training", "B_training_candidate_detection")
        c = self.stage("A_clean_threshold_training", "C_discrepancy_and_explanation_review")
        self.assertEqual("high_training_like_warning", result["final_route"])
        self.assertIn("distributed_training_like_candidate", b["labels"])
        self.assertIn("checkpoint_training_like_candidate", b["labels"])
        self.assertEqual("C2_candidate_conflict_adjudication", c["mode"])
        self.assertFalse(c["discrepancies"])
        self.assertFalse(c["missing_channels"])

    def test_large_compute_alone_does_not_become_medium_or_high_training(self) -> None:
        result = self.result("K_large_compute_alone")
        b = self.stage("K_large_compute_alone", "B_training_candidate_detection")
        self.assertEqual("weak_training_like_candidate", result["final_route"])
        self.assertIn("large_compute_candidate", b["labels"])
        self.assertNotIn("distributed_training_like_candidate", b["labels"])
        self.assertNotIn("checkpoint_training_like_candidate", b["labels"])
        self.assertIn("large_compute_training_identity_unresolved", result["caveats"])

    def test_activity_fabric_and_storage_alone_do_not_become_training(self) -> None:
        cases = {
            "L_activity_alone": "activity alone",
            "M_fabric_alone": "fabric alone",
            "N_storage_writes_alone": "storage writes alone",
        }
        for key, description in cases.items():
            with self.subTest(case=description):
                result = self.result(key)
                b = self.stage(key, "B_training_candidate_detection")
                c = self.stage(key, "C_discrepancy_and_explanation_review")
                self.assertEqual("no_training_like_candidate_detected_in_covered_live_segment", result["final_route"])
                self.assertEqual([], b["labels"])
                self.assertEqual("C1_negative_screen_integrity", c["mode"])
                self.assertIn("negative_screen_coverage_sufficient", c["labels"])

    def test_storage_explanation_demotes_checkpoint_candidate(self) -> None:
        result = self.result("E_storage_operation_explains_checkpoint")
        b = self.stage("E_storage_operation_explains_checkpoint", "B_training_candidate_detection")
        c = self.stage("E_storage_operation_explains_checkpoint", "C_discrepancy_and_explanation_review")
        self.assertIn("checkpoint_training_like_candidate", b["labels"])
        self.assertIn("candidate_explained_by_storage_operation", c["labels"])
        self.assertEqual("candidate_explained_or_demoted", result["final_route"])

    def test_serving_counterevidence_demotes_large_compute_candidate(self) -> None:
        result = self.result("F_serving_inference_counterevidence")
        c = self.stage("F_serving_inference_counterevidence", "C_discrepancy_and_explanation_review")
        self.assertIn("candidate_explained_by_serving", c["labels"])
        self.assertEqual("candidate_explained_or_demoted", result["final_route"])

    def test_benchmark_and_hpc_alternative_demotes_fabric_candidate(self) -> None:
        result = self.result("G_hpc_mpi_benchmark_alternative")
        c = self.stage("G_hpc_mpi_benchmark_alternative", "C_discrepancy_and_explanation_review")
        self.assertIn("candidate_benchmark_like", c["labels"])
        self.assertIn("candidate_hpc_mpi_alternative", c["labels"])
        self.assertEqual("candidate_explained_or_demoted", result["final_route"])

    def test_covered_negative_and_missing_negative_screen_routes(self) -> None:
        covered = self.result("B_covered_negative")
        missing = self.result("D_missingness_blocks_negative_screen")
        self.assertEqual("no_training_like_candidate_detected_in_covered_live_segment", covered["final_route"])
        self.assertIn(
            "negative_screen_coverage_sufficient",
            self.stage("B_covered_negative", "C_discrepancy_and_explanation_review")["labels"],
        )
        self.assertEqual("inconclusive_due_to_missingness", missing["final_route"])
        self.assertIn(
            "negative_screen_blocked_by_missingness",
            self.stage("D_missingness_blocks_negative_screen", "C_discrepancy_and_explanation_review")["labels"],
        )

    def test_capacity_and_activity_attribution_conflicts_route_integrity(self) -> None:
        capacity = self.result("H_capacity_claim_conflict")
        attribution = self.result("I_activity_attribution_conflict")
        self.assertEqual("integrity_review_required", capacity["final_route"])
        self.assertIn(
            "capacity_claim_conflict",
            self.stage("H_capacity_claim_conflict", "C_discrepancy_and_explanation_review")["labels"],
        )
        self.assertEqual("integrity_review_required", attribution["final_route"])
        self.assertIn(
            "activity_attribution_conflict",
            self.stage("I_activity_attribution_conflict", "C_discrepancy_and_explanation_review")["labels"],
        )


if __name__ == "__main__":
    unittest.main()
