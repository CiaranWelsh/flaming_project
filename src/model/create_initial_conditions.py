
import numpy
from lxml import etree
import os
import os, glob


'''    <itno>0</itno>
    <environment></environment>
    <xagent>
        <name>A</name>
        <id>0</id>
        <x>-0.12465215</x>
        <y>-0.2607972</y>
        <z>-0.39554715</z>
        <fx>0.1592794</fx>
        <fy>-0.29218417</fy>
        <fz>-0.29697996</fz>
    </xagent>'''


def create_root():
    root = etree.Element('states')
    itno = etree.SubElement(root, 'itono')
    itno.text = '0'
    etree.SubElement(root, 'environment')
    return root


def add_agent(root, agent_name, id, x, y, z, fx, fy, fz):
    agent = etree.SubElement(root, 'xagent')
    name = etree.SubElement(agent, 'name')
    _id = etree.SubElement(agent, 'id')
    _x = etree.SubElement(agent, 'x')
    _y = etree.SubElement(agent, 'y')
    _z = etree.SubElement(agent, 'z')
    _fx = etree.SubElement(agent, 'fx')
    _fy = etree.SubElement(agent, 'fy')
    _fz = etree.SubElement(agent, 'fz')

    name.text = agent_name
    _id.text = str(id)
    _x.text = str(x)
    _y.text = str(y)
    _z.text = str(z)
    _fx.text = str(fx)
    _fy.text = str(fy)
    _fz.text = str(fz)
    return root


def add_agents(n, root, agent_name):
    ## gen some numbers
    for i in range(n):
        x, y, z = numpy.random.uniform(-1.0, 1.0, 3)
        fx, fy, fz = numpy.random.uniform(-1.0, 1.0, 3)
        add_agent(root, agent_name, i, x, y, z, fx, fy, fz)

    return root


def to_file(root, fname):
    with open(fname, 'w') as f:
        f.write(etree.tostring(root, pretty_print=True, encoding='unicode'))


if __name__ == '__main__':
    root = create_root()
    root = add_agents(10, root, 'A')
    # print(etree.tostring(root, encoding='utf8'))

    fname = os.path.join(r'D:\Documents\FirstFlameProject\flaming_project\flaming_project\bin\x64\Release_Visualisation', 'init.xml')
    to_file(root, fname)



    # add_agent(root, )































