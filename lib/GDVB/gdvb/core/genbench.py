import random
import datetime

from .verification_benchmark import VerificationBenchmark


# main benchmark generation function
def gen(settings):
    start_time = datetime.datetime.now()
    random.seed(settings.seed)

    verification_benchmark = VerificationBenchmark(
        settings.name, settings.dnn_configs, settings.ca_configs, settings
    )

    #  perform tasks
    if settings.task == "C":
        pass
    elif settings.task == "T":
        verification_benchmark.train()
        # verification_benchmark.analyze_training()
    elif settings.task == "P":
        verification_benchmark.gen_props()
    # elif settings.task == "CRA":
    #    verification_benchmark.critical_region_analysis()
    elif settings.task == "V":
        verification_benchmark.verify()
        # verification_benchmark.analyze_verification()
    elif settings.task == "A":
        verification_benchmark.analyze_all()
        verification_benchmark.save_results()
    elif settings.task == "E":
        verification_benchmark.train()
        verification_benchmark.gen_props()
        verification_benchmark.verify()
        verification_benchmark.analyze_all()
        verification_benchmark.save_results()
    else:
        raise Exception("Unknown task.")

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    settings.logger.info(f"Spent {duration} seconds.")
