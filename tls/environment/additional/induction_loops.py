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

        positions = [' Westside intersection ', ' Eastside intersection ',
                     ' Southside intersection ', ' Northside intersection ']
        for position in positions:
            position = trafficlight.findNext(string=position)

            detector_locations = [' Departure detectors ', ' Stopbar detectors ']
            detectors = {}
            for detector_location in detector_locations:
                detectors_ = position.findNext(string=detector_location)

                detector_types = {'inductionLoop': [], 'laneAreaDetector': []}
                next_ = detectors_.next
                while not (next_ is None or isinstance(next_, bs4.Comment)):
                    if next_.name in detector_types:
                        detector_types[next_.name].append(next_.attrs['id'])
                    next_ = next_.next
                detectors[detector_location] = detector_types
            intersections[position] = detectors

        induction_loops[trafficlight_id] = intersections
    return induction_loops


if __name__ == '__main__':
    r = process_additional_file('/home/gosha/workspace/pycharm/adaptive-tls/'
                                'networks/montgomery_county/moco.det.xml')
    print(r['cluster_648538736_648538737'][' Westside intersection '][' Stopbar detectors '])
