from typing import Optional
from urllib.parse import urlparse
import argparse
from dataclasses import dataclass


@dataclass
class Args:
    url_file: str
    domain_save_file: Optional[str]


def get_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, help="File containing urls")
    parser.add_argument("--domain-save-file", "-s", type=str, required=False, help="Optional file to store all unique domains")

    parsed = parser.parse_args()

    return Args(
        url_file=parsed.file,
        domain_save_file=parsed.domain_save_file
    )

def extract_unique_domains(urls: list[str]) -> set[str]:
    domains = set()

    for url in urls:
        parsed = urlparse(url)
        domains.add(f"{parsed.scheme}://{parsed.netloc}")
    
    try:
        domains.remove("://")
    except KeyError:
        pass
    
    return domains

def main(args: Args):
    with open(args.url_file, "r") as f:
        urls = [u[:-1] if u.endswith("\n") else u for u in f.readlines()]

    domains = extract_unique_domains(urls)

    sf = args.domain_save_file

    if sf is not None:
        with open(sf, "w") as f:
            for d in sorted(domains):
                f.write(f"{d}\n")

    print(f"Amount of unique domains: {len(domains)}")


if __name__ == "__main__":
    main(get_args())
