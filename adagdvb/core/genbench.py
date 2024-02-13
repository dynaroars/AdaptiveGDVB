import random
import datetime

from gdvb.core.verification_benchmark import VerificationBenchmark
from ..core.evo_bench import EvoBench


def generate(settings):
    start_time = datetime.datetime.now()
    random.seed(settings.seed)

    verification_benchmark = VerificationBenchmark(
        settings.name, settings.dnn_configs, settings.ca_configs, settings
    )

    evo_bench = EvoBench(verification_benchmark)
    evo_bench.run()

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    settings.logger.info(f"Spent {duration} seconds.")
