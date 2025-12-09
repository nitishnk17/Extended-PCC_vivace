"""Quick essential tests for FairnessController"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.fairness_controller import FairnessController, FairnessState
from src.contention_detector import ContentionLevel
from src.config import Config

class TestBasics:
    @pytest.fixture
    def controller(self):
        return FairnessController(Config().fairness_controller)

    def test_init(self, controller):
        assert controller.mu == 0.5
        assert controller.estimated_fair_share is None

    def test_fair_share_aggregate(self, controller):
        controller.estimate_fair_share(10.0, 3, 30.0)
        assert controller.estimated_fair_share == pytest.approx(10.0, abs=0.1)

    def test_augment_utility_penalty(self, controller):
        controller.estimate_fair_share(10.0, 3, 30.0)
        utility_aug = controller.augment_utility(50.0, 15.0, ContentionLevel.MODERATE, 1.0)
        assert utility_aug < 50.0  # Penalty applied

    def test_adaptive_mu_solo(self, controller):
        assert controller._get_adaptive_mu(ContentionLevel.SOLO) == 0.0

    def test_adaptive_mu_heavy(self, controller):
        mu_heavy = controller._get_adaptive_mu(ContentionLevel.HEAVY)
        assert mu_heavy == pytest.approx(0.75, abs=0.01)  # 1.5 * 0.5

    def test_fairness_ratio(self, controller):
        controller.estimated_fair_share = 10.0
        controller.current_rate = 10.0
        assert controller.get_fairness_ratio() == pytest.approx(1.0)

    def test_above_fair_share(self, controller):
        controller.estimated_fair_share = 10.0
        controller.current_rate = 15.0
        assert controller.is_above_fair_share() == True

    def test_statistics(self, controller):
        controller.estimate_fair_share(10.0, 3, 30.0)
        controller.augment_utility(50.0, 15.0, ContentionLevel.MODERATE, 1.0)
        stats = controller.get_statistics()
        assert stats['current']['fair_share'] == pytest.approx(10.0, abs=0.1)
        assert stats['statistics']['total_augmentations'] == 1

    def test_reset(self, controller):
        controller.estimate_fair_share(10.0, 3, 30.0)
        controller.augment_utility(50.0, 15.0, ContentionLevel.MODERATE, 1.0)
        controller.reset()
        assert controller.estimated_fair_share is None
        assert controller.total_augmentations == 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
