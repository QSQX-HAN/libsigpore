#!/usr/bin/env python3

# 16th Mar 2023
# modify roland's svg script
# change the nt color to colormap instead of just 3 colors
# change the circle radius
# change the CG and AT pairing, add small circle in the middle for AT pairs
# added margin to avoid edge cut off
# read in offset from filename
# positions are recorded with python idx i.e. start with 0, and correct with offset when display.

# write script into function and iterate. so just go anaconda prompt and run this python script locally.

import sys
import numpy as np
import re
import matplotlib
from cairosvg import svg2png
import matplotlib.cm
import matplotlib.colors

import argparse

def parse_svg(f_svg):
    nuc_positions, nuc_type = [], []
    pos_position, pos_count = [], []
    with open(f_svg, "r") as f:
        for line in f:
            if "</text>" in line:
                if "rgb(0%, 0%, 0%)" in line:
                    row = line.split(">")[1].split("<")[0].replace("T","U")
                    if len(row) == 1:
                        nuc_type.append(row)
                        x = float(line.split('x="')[1].split('"')[0])
                        y = float(line.split('y="')[1].split('"')[0])
                        nuc_positions.append([x, y])
                else:
                    posmark = int(line.split(">")[1].split("<")[0])
                    x = float(line.split('x="')[1].split('"')[0])
                    y = float(line.split('y="')[1].split('"')[0])
                    pos_position.append([x, y])
                    pos_count.append(posmark) # no need offset since the svh alrd has offset coordinates

    return nuc_positions, nuc_type, pos_position, pos_count

def parse_ct(f_ct):
    # expect 1-based coordinates
    pairs = []
    with open(f_ct, "r") as f:
        for no, line in enumerate(f):
            if no == 0:
                continue

            row = line.split()
            a, b = int(row[0]), int(row[4])
            if b != 0:
                pairs.append([a-1, b-1])
    return pairs

def parse_shape(f_shape):
    shape = []
    with open(f_shape, "r") as f:
        for line in f:
            row = line.split()
            shape.append(float(row[1]))
    print(shape)
    return shape

def parse_marker(f_markers, offset=0):
    # parsing marker
    marker_colors={
        'P1':'rgb(55%,15%,80%)',
        'P2':'rgb(100%,0%,80%)',
        'P3':'rgb(100%,0%,20%)',
        'P45':'rgb(100%,40%,0%)',
        'P6':'rgb(40%,0%,100%)',
        'P7':'rgb(60%,100%,0%)',
        'P8':'rgb(0%,100%,60%)',
        'P9':'rgb(0%,100%,100%)',
    }

    markers={
        'P1':[],
        'P2':[],
        'P3':[],
        'P45':[],
        'P6':[],
        'P7':[],
        'P8':[],
        'P9':[],
    }
    
    f=open(sys.argv[4]).readlines()
    with open(f_markers, "r") as f:
        for line in f:
            row = line.strip().split()
            marker_type = row[0]
            marker_pos = int(l[1])-offset
            markers[marker_type].append(marker_pos)
        
    return markers


def shape_to_color_v1(shape):
    if shape < -998:
        return "rgb(100%, 100%, 100%)"
    else:
        if shape < 0:
            return "rgb(90%, 90%, 90%)"
        elif shape > 4:
            return "rgb(90%, 0%, 0%)"
        else:
            val = int((4-shape)/4.0*90)
            return "rgb(90%%, %d%%, %d%%)"%(val, val)

def shape_to_color_v2(shape):
    if shape < -998:
        return "rgb(100%, 100%, 100%)"
    else:
        if shape < 0.35:
            return "rgb(80%, 80%, 80%)"
        elif shape < 0.85:
            return "rgb(90%, 49%, 13%)"
        else:
            return "rgb(90%, 30%, 23%)"


def append_header(list_svg_output, span_x, span_y, margin):
    
    list_svg_output.append('''<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" 
    "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
    ''')

    list_svg_output.append('<svg width="%f" height="%f" version="1.1" xmlns="http://www.w3.org/2000/svg">'%(span_x + 2*margin, span_y + 2*margin))

def append_bp_line(list_svg_output, pairs, nuc_positions, margin, nuc_type, pair_circle_radius):

    # draw base pairing line
    for p in pairs:
        pos1, pos2 = p[0], p[1]  # alrd corrected to zero based
        x1 = nuc_positions[p[0]][0] + margin
        y1 = nuc_positions[p[0]][1] + margin
        x2 = nuc_positions[p[1]][0] + margin
        y2 = nuc_positions[p[1]][1] + margin
        if not ((nuc_type[pos1]=="G" or nuc_type[pos1]=="C") and (nuc_type[pos2]=="G" or nuc_type[pos2]=="C")): # draw a circle to note GC pairing
            list_svg_output.append('<line x1="%f" y1="%f" x2="%f" y2="%f" stroke="rgb(0%%, 0%%, 0%%)" stroke-width="1.0" />'%(x1,y1,x2,y2))
            list_svg_output.append('<circle cx="%f" cy="%f" r="%f" stroke="rgb(0%%, 0%%, 0%%)" stroke-width="1.0" fill="rgb(0%%, 0%%, 0%%)" />'%((x1+x2)/2, (y1+y2)/2, pair_circle_radius))
        else:
            list_svg_output.append('<line x1="%f" y1="%f" x2="%f" y2="%f" stroke="rgb(0%%, 0%%, 0%%)" stroke-width="1.0" />'%(x1,y1,x2,y2))

def append_nt_line(list_svg_output, nuc_positions, margin):

    # draw adjacent nt line
    for n in range(len(nuc_positions)-1): 
        x1 = nuc_positions[n][0] + margin
        y1 = nuc_positions[n][1] + margin
        x2 = nuc_positions[n+1][0] + margin
        y2 = nuc_positions[n+1][1] + margin
        list_svg_output.append('<line x1="%f" y1="%f" x2="%f" y2="%f" stroke="rgb(0%%, 0%%, 0%%)" stroke-width="1.0" />'%(x1,y1,x2,y2))

def append_circles(list_svg_output, nuc_positions, nuc_type, nuc_color, margin, nt_circle_radius):
    # draw circles
    for n, nuc in enumerate(nuc_type):
        x = nuc_positions[n][0] + margin
        y = nuc_positions[n][1] + margin
        list_svg_output.append('<circle cx="%f" cy="%f" r="%f" stroke="None" fill="%s"/>'%(x, y, nt_circle_radius, nuc_color[n]))
        list_svg_output.append('<text x="%f" y="%f" text-anchor="middle" font-family="Verdana" font-size="10.5" >%s</text>'%(x,y+3.5,nuc))

def append_positions(list_svg_output, pos_count, pos_position, margin):
    # draw position labels
    for p, pos in enumerate(pos_count):
        x = pos_position[p][0] + margin
        y = pos_position[p][1] + margin
        list_svg_output.append('<text x="%f" y="%f" text-anchor="end" font-family="Verdana" font-size="7.5" >%s</text>'%(x, y, pos)) # change anchor to "end"

def append_footer(list_svg_output):
    list_svg_output.append('</svg>')

def annotate_svg(f_svg, pairs, shape, f_out_svg, p_colormap=True):

    nuc_positions, nuc_type, pos_position, pos_count = parse_svg(f_svg)

    if not p_colormap:
        nuc_color = [ shape_to_color_v1(s) for s in shape]
    else:
        cmap = matplotlib.cm.get_cmap('Reds')
        shape_nona = [x for x in shape if x!=-999]
        if shape_nona==[]:
            nuc_color=[(100,100,100)]*len(shape)
        else:
            print(shape_nona)
            norm = matplotlib.colors.Normalize(vmin=min(shape_nona)-0.3, vmax=max(shape_nona)+1) # vmax is black, offset a bit
            nuc_color = list(map(lambda x: cmap(norm(x)) if x!=-999 else (100,100,100), shape)) # scale only non NA shape, if NA make it white
        nuc_color = list(map(lambda tup: f"rgb({tup[0]*100}%, {tup[1]*100}%, {tup[2]*100}%)", nuc_color)) # render to string


    # get maximum and minimum of x and y to define svg size
    nuc_positions = np.array(nuc_positions)
    min_x = min(nuc_positions[:,0])
    max_x = max(nuc_positions[:,0])
    min_y = min(nuc_positions[:,1])
    max_y = max(nuc_positions[:,1])

    span_x = abs(max_x-min_x)
    span_y = abs(max_y-min_y)

    for n in range(len(nuc_positions)):
        nuc_positions[n] = nuc_positions[n]-np.array(min_x,min_y)
        

    # draw setting
    nt_circle_radius = 6
    pair_circle_radius = 2.75 #same as varna, for the AT pairing
    margin = 50 # the edges of the svg always get cutoff. add margin to avoid this. 

    list_svg_output = []
    append_header(list_svg_output, span_x, span_y, margin)
    append_bp_line(list_svg_output, pairs, nuc_positions, margin, nuc_type, pair_circle_radius)
    append_nt_line(list_svg_output, nuc_positions, margin)
    append_circles(list_svg_output, nuc_positions, nuc_type, nuc_color, margin, nt_circle_radius)
    append_positions(list_svg_output, pos_count, pos_position, margin)
    append_footer(list_svg_output)

    with open(f_out_svg, "w") as f:
        for line in list_svg_output:
            f.write("%s\n" % line)
    

def annotate_svg_cmdline(f_svg, f_ct, f_shape, f_out_svg, p_colormap=True):
    pairs = parse_ct(f_ct)
    shape = parse_shape(f_shape)
    annotate_svg(f_svg, pairs, shape, f_out_svg, p_colormap)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", default=None, help="input svg file")
    parser.add_argument("-c", default=None, help="ct file")
    parser.add_argument("-s", default=None, help="shape file")
    parser.add_argument("-o", default=None, help="output: svg file")
    parser.add_argument("-p", default=None, help="output: png file")

    args = parser.parse_args()
    f_svg = args.i
    f_ct = args.c
    f_shape = args.s
    f_output_svg = args.o
    #f_output_png = args.p

    annotate_svg_cmdline(f_svg, f_ct, f_shape, f_output_svg)

    # convert svg to png        
    #svg2png(url=f_out_svg, write_to=f_out_png)



if __name__ == "__main__":
    main()


