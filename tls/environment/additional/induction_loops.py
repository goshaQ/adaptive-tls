import re
import bs4


def process_additional_file(path):
    with open(path, 'r') as f:
        soup = bs4.BeautifulSoup(f, features='xml')
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
            detectors = {}
            for detector_type in detector_types:
                detectors_ = position.findNext(string=detector_type)

                detectors[detector_type] = []
                next_ = detectors_.next
                while not (next_ is None or isinstance(next_, bs4.Comment)):
                    if next_.name == 'inductionLoop':
                        detectors[detector_type].append(next_.attrs['id'])
                    next_ = next_.next
            intersections[position_type] = detectors

        induction_loops[trafficlight_id] = intersections
    return induction_loops
