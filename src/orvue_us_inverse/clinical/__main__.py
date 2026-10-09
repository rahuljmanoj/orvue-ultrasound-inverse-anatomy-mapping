"""python -m orvue_us_inverse.clinical [--tab bmode | mapping] [--case X] [--no-camera]: the clinical window."""
from orvue_us_inverse.clinical.app import main

if __name__ == "__main__":
    raise SystemExit(main())
