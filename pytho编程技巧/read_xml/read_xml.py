import xml.etree.ElementTree as ET 
tree = ET.parse('./test.xml')
root = tree.getroot()

print(root.tag)

for child in root:
    print(child.tag,child.attrib)

print(root[0][1].text)

for neighbor in root.iter('neighbor'):
    print(neighbor.attrib)