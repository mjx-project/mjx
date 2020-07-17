import xml.etree.ElementTree as ET
import urllib.parse


if __name__ == "__main__":
    tree = ET.parse("resources/2011020417gm-00a9-0000-b67fcaa3.mjlog")
    root = tree.getroot()
    print(root.tag)
    print(root.attrib)
    for child in root:
        print(child.tag)
        print(child.attrib)

        if child.tag == "UN":
            print(urllib.parse.unquote(child.attrib["n0"]))
            print(urllib.parse.unquote(child.attrib["n1"]))
            print(urllib.parse.unquote(child.attrib["n2"]))
            print(urllib.parse.unquote(child.attrib["n3"]))
            break
