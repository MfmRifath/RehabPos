"""Regenerate reports/samples/sample_report.{json,pdf} from a SYNTHETIC session."""

from rehabpose.api.samples import write_sample_report
from rehabpose.core.exercise import load_exercises
from rehabpose.settings import get_settings, project_root

if __name__ == "__main__":
    settings = get_settings()
    exercise = load_exercises(settings.exercises_dir)["shoulder_flexion"]
    for path in write_sample_report(exercise, project_root() / "reports" / "samples"):
        print(path)
