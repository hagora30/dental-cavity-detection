import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImports(unittest.TestCase):
    """Verify all pipeline dependencies import correctly."""

    def test_ultralytics(self):
        import ultralytics
        self.assertIsNotNone(ultralytics.__version__)
        print(f"   ultralytics {ultralytics.__version__}")

    def test_cv2(self):
        import cv2
        self.assertIsNotNone(cv2.__version__)
        print(f"   opencv {cv2.__version__}")

    def test_albumentations(self):
        import albumentations
        self.assertIsNotNone(albumentations.__version__)
        print(f"   albumentations {albumentations.__version__}")

    def test_wandb(self):
        import wandb
        self.assertIsNotNone(wandb.__version__)
        print(f"   wandb {wandb.__version__}")

    def test_roboflow(self):
        import roboflow
        self.assertIsNotNone(roboflow.__version__)
        print(f"  roboflow {roboflow.__version__}")

    def test_yaml(self):
        import yaml
        self.assertIsNotNone(yaml.__version__)
        print(f"   PyYAML {yaml.__version__}")

    def test_dotenv(self):
        from dotenv import load_dotenv
        self.assertIsNotNone(load_dotenv)
        print(f"   python-dotenv")


class TestProjectStructure(unittest.TestCase):
    """Verify the expected project folder structure exists."""

    def setUp(self):
        self.root = Path(__file__).parent.parent

    def test_configs_exist(self):
        self.assertTrue((self.root / "configs" / "dataset.yaml").exists())
        self.assertTrue((self.root / "configs" / "train_config.yaml").exists())
        print("   configs/dataset.yaml and train_config.yaml exist")

    def test_src_scripts_exist(self):
        scripts = [
            "data_ingest.py",
            "data_cleaning.py",
            "rebuild_splits.py",
            "eda.py",
            "augmentation.py",
            "train.py",
            "evaluate.py",
            "predict.py",
        ]
        for script in scripts:
            path = self.root / "src" / script
            self.assertTrue(path.exists(), f"Missing: src/{script}")
        print(f"   All {len(scripts)} src scripts present")

    def test_requirements_exist(self):
        self.assertTrue((self.root / "requirements.txt").exists())
        self.assertTrue((self.root / "requirements-cloud.txt").exists())
        print("   requirements.txt and requirements-cloud.txt exist")

    def test_gitignore_blocks_data(self):
        gitignore = (self.root / ".gitignore").read_text()
        self.assertIn("data/", gitignore)
        self.assertIn(".env", gitignore)
        self.assertIn("*.pt", gitignore)
        print("   .gitignore blocks data/, .env, and *.pt")


class TestDatasetYaml(unittest.TestCase):
    """Verify dataset.yaml is valid and well-formed."""

    def setUp(self):
        import yaml
        self.root = Path(__file__).parent.parent
        yaml_path = self.root / "configs" / "dataset.yaml"
        with open(yaml_path) as f:
            self.cfg = yaml.safe_load(f)

    def test_has_required_keys(self):
        for key in ["path", "train", "val", "nc", "names"]:
            self.assertIn(key, self.cfg, f"Missing key: {key}")
        print("   dataset.yaml has all required keys")

    def test_class_count(self):
        self.assertEqual(self.cfg["nc"], 4)
        self.assertEqual(len(self.cfg["names"]), 4)
        print(f"   nc=4, names={self.cfg['names']}")

    def test_class_names(self):
        expected = ["Cavity", "Fillings", "Impacted Tooth", "Implant"]
        self.assertEqual(self.cfg["names"], expected)
        print("   Class names match expected")


class TestTrainConfig(unittest.TestCase):
    """Verify train_config.yaml is valid and well-formed."""

    def setUp(self):
        import yaml
        self.root = Path(__file__).parent.parent
        yaml_path = self.root / "configs" / "train_config.yaml"
        with open(yaml_path) as f:
            self.cfg = yaml.safe_load(f)

    def test_has_required_sections(self):
        for section in ["model", "training", "augmentation", "logging", "data"]:
            self.assertIn(section, self.cfg, f"Missing section: {section}")
        print("   train_config.yaml has all required sections")

    def test_model_architecture(self):
        arch = self.cfg["model"]["architecture"]
        self.assertIn("yolov8", arch)
        print(f"   model architecture: {arch}")

    def test_hsv_disabled_for_xray(self):
        aug = self.cfg["augmentation"]
        self.assertEqual(aug["hsv_h"], 0.0, "hsv_h should be 0 for X-rays")
        self.assertEqual(aug["hsv_s"], 0.0, "hsv_s should be 0 for X-rays")
        print("   HSV hue/saturation disabled for grayscale X-rays")

    def test_flipud_disabled(self):
        self.assertEqual(self.cfg["augmentation"]["flipud"], 0.0)
        print("   Vertical flip disabled — upside-down jaw is invalid")


class TestProcessedDataExists(unittest.TestCase):
    """Verify processed dataset exists and has expected structure."""

    def setUp(self):
        self.root = Path(__file__).parent.parent
        self.processed = self.root / "data" / "processed"

    def test_splits_exist(self):
        for split in ["train", "valid", "test"]:
            img_dir = self.processed / split / "images"
            lbl_dir = self.processed / split / "labels"
            self.assertTrue(img_dir.exists(), f"Missing: {img_dir}")
            self.assertTrue(lbl_dir.exists(), f"Missing: {lbl_dir}")
        print("   train/valid/test splits exist in data/processed/")

    def test_train_has_enough_images(self):
        train_imgs = list(
            (self.processed / "train" / "images").glob("*")
        )
        self.assertGreater(len(train_imgs), 800)
        print(f"   train split has {len(train_imgs)} images")

    def test_images_match_labels(self):
        for split in ["train", "valid", "test"]:
            imgs = set(
                f.stem for f in
                (self.processed / split / "images").glob("*")
            )
            lbls = set(
                f.stem for f in
                (self.processed / split / "labels").glob("*.txt")
            )
            unmatched = imgs - lbls
            self.assertEqual(
                len(unmatched), 0,
                f"{split}: {len(unmatched)} images without labels"
            )
        print("   All images have matching label files")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  DENTAL CAVITY DETECTION — PIPELINE SMOKE TESTS")
    print("=" * 55 + "\n")

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # Run in logical order
    for cls in [
        TestImports,
        TestProjectStructure,
        TestDatasetYaml,
        TestTrainConfig,
        TestProcessedDataExists,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    print("\n" + "=" * 55)
    if result.wasSuccessful():
        print(f"   All {result.testsRun} tests passed")
    else:
        print(f"   {len(result.failures)} failures, "
              f"{len(result.errors)} errors")
    print("=" * 55 + "\n")
