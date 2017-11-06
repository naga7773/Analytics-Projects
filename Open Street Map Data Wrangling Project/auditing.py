# This file contains the data auditing functions
# import all the required libraries
# re module is used to create Regex expressions
# schema.py includes the schema for validating the output csv files
# and is included in a separate file
# "c" implimentation of the Element Tree module is being used here
# to increase the speed.

import csv
import codecs
import re
import xml.etree.cElementTree as ET
from collections import defaultdict
from collections import Counter
from collections import OrderedDict
import cerberus
import pprint
import schema
import os
import itertools

# I'm using Bryan- College Station- Nevasota map in xml format.
samplefile = "bcs_area"

# Get the size of the file

print " The size of the file in bytes : ",os.path.getsize(samplefile)

# Get a list of the number of elements of each type

def element_type_list():
    tagsdict={}
    for event,element in ET.iterparse(samplefile):
        if element.tag in tagsdict.keys():
            tagsdict[element.tag] +=1
        else:
            tagsdict[element.tag] =1
    return tagsdict

print " The number of occurences of different element types :\n",element_type_list()

# finding different "k" types in tag elements.
# creating a list of kvalues (All values - repeated)


kvalues = []
for event, elem in ET.iterparse(samplefile):
    for element in elem:
        if element.tag =="tag":
            kvalues.append(element.attrib['k'])
            
# Counting the number of occurences of each k type
# using "Counter" from "collections"
# creating a dictionary where the keys are the diffrent k types and the corresponding values represent the number of occurences.

kvaluedict = Counter(kvalues)
keys = kvaluedict.keys()
values = kvaluedict.values()
def top_10_ktypes():
    kvaluesdiscend = OrderedDict(sorted(kvaluedict.items(), key=lambda t:t[1]))
    x = itertools.islice(kvaluesdiscend.items(), (len(kvaluesdiscend.keys())-10), len(kvaluesdiscend.keys()))

    for key, value in x:
        print key, value
        

print " The top 10 number of occurences of each k type is : \n", top_10_ktypes()

# Finding out the k-type that occured more number of times.

print " The k-type that occured more number of times is :",keys[values.index(max(values))]
print " It occured for :",values[values.index(max(values))]

# Auditing Street types

# defining a reg ex expression.
# This expression matches at the end

street_types_re = re.compile(r'\b\S+\.?$',re.IGNORECASE)

# This dictionary has the street type as the key and the set of street names having that particular street type as the value of the key.
street_types = defaultdict(set)

# expected street types
expected =["Avenue","Road","Boulevard","Street","Court","Circle","Place","Drive","Lane"]

# This function checks if the tag value is a street name and returns a Boolean value
def is_street_name(element):
    return (element.attrib['k']=="addr:street" )

# This function takes the strret name and finds the reg ex match to find out the street type.
# If the street type is not in the expected list, Then it adds the street type to "street_types" dictionary
def audit_street_type(street_types,street_name):
    m =street_types_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)
def audit():
    for event,element in ET.iterparse(samplefile,events=("start",)):
        if element.tag == "way" or element.tag == 'node':
            for tag in element.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types,tag.attrib['v'])
                    
    pprint.pprint(dict(street_types))

# Let us check the street types now.

audit()

# There are some abbrveiations for the street types.

# Let us rename those street names so that a common notation is followed.

# FIrst create the mapping dictionry for required conversions.

mapping = { "Rd":"Road","S":"South","Ave.":"Avenue"}

# This function updates the name of the street.
def update_name(name, mapping):
    m = street_types_re.search(name)
    n =m.group()
    if n in mapping.keys():
        #print "before changing : ",n
        name = street_types_re.sub(mapping[n],name)
        #print "changing name to : ",name
        return name  
    else:
        return name
# Let us check how the update_name function works.
# 'S Texas Ave.' is taken from the previous audit() result.
old_name = 'S Texas Ave.'
new_name = update_name(old_name,mapping)
print "Old name is : ",old_name
print "New name is : ",new_name

## Auditing postal codes

def postal_codes():
    postcode_list =[]
    for event,element in ET.iterparse(samplefile):
        for elem in element.iter("tag"):
            if elem.attrib['k'] == "addr:postcode":
                postcode_list.append(elem.attrib['v'])
    pprint.pprint(Counter(postcode_list))

postal_codes()

# This indicates that there is no consistency in the postal codes format used in the maps file.
# There is a prefix TX for some of the reocrds.
# Let us clean it.
def update_postalcodes(postalcode):
    if len(postalcode) == 5:
        return postalcode
    else:
        postalcode = postalcode[3:]
        return postalcode

        
# finding the problemetic k-types

# This finds if the letters are all lowercase letters
lower = re.compile(r'^([a-z]|_)*$')

# This checks if there is a semi colon in the k-type
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')

# This checks if there is a problematic character
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

def key_type(element, keys):
    if element.tag == "tag":
        if lower.search(element.attrib['k']):
            keys['lower'] +=1
        elif lower_colon.search(element.attrib['k']):
            keys['lower_colon'] += 1
        elif problemchars.search(element.attrib['k']):
            keys['problemchars'] +=1
        else:
            keys['other'] +=1
        
    return keys

def keys_map(samplefile):
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(samplefile):
        keys = key_type(element, keys)

    return keys

print"Different type of K after subdividing : ",keys_map(samplefile)

OSM_PATH = samplefile
NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

SCHEMA = schema.schema
# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']


def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""
    
    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags_node = []  # Handle secondary tags the same way for both node and way elements
    tags_ways=[]
    if element.tag == 'node':
        for i in NODE_FIELDS:
            node_attribs[i]=element.attrib[i]
        #for i in NODE_TAGS_FIELDS:
            
        for x in element.getchildren():
            #print x.attrib['k']
            dicty ={}
            m = LOWER_COLON.search(x.attrib['k'])
            n = PROBLEMCHARS.search(x.attrib['k'])
            if m:
                xx =x.attrib['k'].split(':')
                if len(xx) ==3:
                    dicty['key'] = xx[1]+":"+xx[2]
                else:
                    dicty['key'] = xx[1]
                dicty['type'] = xx[0]
                dicty['value']=x.attrib['v']
            elif n:
                continue
            else:
                dicty['key'] = x.attrib['k']
                dicty['type'] = default_tag_type
                dicty['value'] = x.attrib['v']
            
            dicty['id'] = element.attrib['id']
          
                    
            tags_node.append(dicty)
        return {'node': node_attribs, 'node_tags': tags_node}
    elif element.tag == 'way':
        
        for i in WAY_FIELDS:
            way_attribs[i] = element.attrib[i]
        count=0    
        for y in element.getchildren():
            
            
            if y.tag=='tag':
                victy ={}
                m = LOWER_COLON.search(y.attrib['k'])
                n = PROBLEMCHARS.search(y.attrib['k'])
                if m:
                    xx =y.attrib['k'].split(':')
                    if len(xx) ==3:
                        victy['key'] = xx[1]+":"+xx[2]
                    else:
                        victy['key'] = xx[1]
                    victy['type'] = xx[0]
                    victy['value']=y.attrib['v']
                elif n:
                    continue
                else:
                    victy['key'] = y.attrib['k']
                    victy['type'] = default_tag_type
                    victy['value'] = y.attrib['v']
            
                victy['id'] = element.attrib['id']
                tags_ways.append(victy)
                
            else:
                kitty={}
                   
                if y.tag =='nd':
                    kitty['id'] = element.attrib['id']
                    kitty['node_id']=y.attrib['ref']
                    kitty['position']= count
                    count +=1
                #print kitty
                way_nodes.append(kitty)
    
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags_ways}
    else:
        return None

# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    tag.attrib['v'] = update_name(tag.attrib['v'],mapping)
                if tag.attrib['k'] == "addr:postcode":
                    tag.attrib['v'] = update_postalcodes(tag.attrib['v'])
            
            yield elem
            root.clear()


def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_strings = (
            "{0}: {1}".format(k, v if isinstance(v, str) else ", ".join(v))
            for k, v in errors.iteritems()
        )
        raise cerberus.ValidationError(
            message_string.format(field, "\n".join(error_strings))
        )


class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS,lineterminator='\n')
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS,lineterminator='\n')
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS,lineterminator='\n')
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS,lineterminator='\n')
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS,lineterminator='\n')

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])



process_map(OSM_PATH, validate=True)

