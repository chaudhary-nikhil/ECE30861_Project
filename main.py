from url import Url
import sys

def parseUrlFile(urlFile: str) -> list[Url]:
    f = open(urlFile, "r")
    url_list : list[Url] = list()

    link_list : list[str] = f.read().split("\n")
    for link in link_list:
        if link == '': # Empty link ie. empty line
            continue
        url_list.append(Url(link))
    f.close()
    return url_list


def main() -> int:
    if (len(sys.argv)) != 2:
        print("URL_FILE is a required argument.")
        return 1

    urlFile = sys.argv[1]
    urls : list[Url] = parseUrlFile(urlFile)
    for url in urls:
        print(url)

    return 0

if __name__ == '__main__':
    return_code: int = main()
