from detector import ObjectDetector


def test_get_detections_returns_empty_list_for_no_boxes():
    class EmptyResult:
        boxes = []

    assert ObjectDetector._get_detections(EmptyResult()) == []
