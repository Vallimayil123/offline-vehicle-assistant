
from app import get_project_status


def test_project_status():
    status = get_project_status()

    assert status["project"] == "Offline AI Vehicle Assistant"
    assert status["project_folder_exists"] is True
    assert status["manuals_folder_exists"] is True
    assert status["python_version"]


def test_python_executable_is_available():
    status = get_project_status()

    assert status["python_executable"]
