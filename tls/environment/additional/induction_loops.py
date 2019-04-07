import re
from bs4 import (
    BeautifulSoup,
    Comment,
)


def process_additional_file(path):
    with open(path, 'r') as f:
        soup = BeautifulSoup(f, features='xml')
    induction_loops = {}

    pattern = re.compile(r'<tlLogic id="(.+?)"')
    trafficlights = soup.findAll(text=pattern)
    for trafficlight in trafficlights:
        trafficlight_id = re.search(pattern, trafficlight).group(1)
        intersections = {}

        position_types = [' Westside intersection ', ' Eastside intersection ',
                          ' Southside intersection ', ' Northside intersection ']
        for position_type in position_types:
            position = trafficlight.findNext(string=position_type)

            detector_types = [' Departure detectors ']
            for detector_type in detector_types:
                detectors = position.findNext(string=detector_type)

                detectors_list = []
                next_ = detectors.next
                while not (next_ is None or isinstance(next_, Comment)):
                    if next_.name == 'inductionLoop':
                        detectors_list.append(next_.attrs['id'])
                    next_ = next_.next
                intersections[position_type] = detectors_list

        induction_loops[trafficlight_id] = intersections
    return induction_loops
