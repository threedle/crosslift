import draccus
from crosslift.pipeline import run
from crosslift.config import Config

@draccus.wrap()
def main(cfg: Config):
    run(cfg)

if __name__ == "__main__":
    main()