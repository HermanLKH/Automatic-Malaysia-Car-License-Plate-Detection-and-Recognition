from roboflow import Roboflow


rf = Roboflow(api_key="FiU3mGWvliwkxuEHra5I")
project = rf.workspace("intelligent-systems-group-8-2025").project("vehicle-license-plate-detection-zasj3")
version = project.version(7)
dataset = version.download("yolov8")

rf = Roboflow(api_key="FiU3mGWvliwkxuEHra5I")
project = rf.workspace("intelligent-systems-group-8-2025").project("vehicle-license-plate-detection-zasj3")
version = project.version(8)
dataset = version.download("yolov8")