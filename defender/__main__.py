import envparse
from apps import create_app

# CUSTOMIZE: import model to be used
import typing
from learning.ensemble.Models.Ensemble import Ensemble
from learning.statefuldefense.predators.DuplicateSimpleScanner import DuplicateSimpleScanner
from learning.statefuldefense.predators.SlackSpaceScanner import SlackSpaceScanner
from learning.statefuldefense.predators.OverlayScanner import OverlayScanner
from learning.statefuldefense.predators.PredatorInterface import PredatorInterface
from learning.statefuldefense.stateful.StatefulDefense import StatefulDefense
from learning.statefuldefense.stateful.StatefulAnnoyDefense import StatefulAnnoyDefense


if __name__ == "__main__":
    # retrieve config values from environment variables
    cfg_file = envparse.env("MODEL_CONFIG_FILE", cast=str, default="")  # TODO default

    # CUSTOMIZE: app and model instance
    model = Ensemble.from_yaml(cfg_file)

    predScanners: typing.List[PredatorInterface] = []
    predScanners.append(DuplicateSimpleScanner(min_number_matches=2, verbose=False))
    predScanners.append(SlackSpaceScanner(min_number_absolute_matches=3,
                                          min_number_relative_matches=0.333334, verbose=False))
    predScanners.append(OverlayScanner(min_ratio=0.75, verbose=False))

    statefulDefense: StatefulDefense = StatefulAnnoyDefense(seed=4878, verbose=False)

    app = create_app(model, predScanners, statefulDefense)

    import sys
    port = int(sys.argv[1]) if len(sys.argv) == 2 else 8080

    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()

    # curl -XPOST --data-binary @somePEfile http://127.0.0.1:8080/ -H "Content-Type: application/octet-stream"
