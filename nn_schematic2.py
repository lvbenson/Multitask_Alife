<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:08:55 2020

@author: benso
"""

from graphviz import Source
temp = """
digraph G {
    rankdir = LR;
    splines=false;
    edge[style=invis];
    ranksep= 1.4;
    {
    node [shape=circle, color=yellow, style=filled, fillcolor=yellow]; #IP Input
    IPI1 [label=<cos(theta)>];
    IPI2 [label=<sin(theta)>]; 
    IPI3 [label=<Theta dot>];
    }
    {
    node [shape=circle, color=coral1, style=filled, fillcolor=coral1]; #CP Input
    CPI1 [label=<     X     >];
    CPI2 [label=<   X dot  >]; 
    CPI3 [label=<   Theta   >];
    CPI4 [label=<Theta dot>];
    }
    {
    node [shape=circle, color=chartreuse, style=filled, fillcolor=chartreuse]; #LW Input
    LWI1 [label=<Theta>];
    LWI2 [label=<Omega>]; 
    LWI3 [label=<LW<Footstate>];
}
{
    node [shape=circle, color=dodgerblue, style=filled, fillcolor=dodgerblue]; #HL1
    HL11 [label=<HL<sub>1</sub><sup>(1)</sup>>];
    HL12 [label=<HL<sub>1</sub><sup>(2)</sup>>];
    HL13 [label=<HL<sub>1</sub><sup>(3)</sup>>];
    HL14 [label=<HL<sub>1</sub><sup>(4)</sup>>];
    HL15 [label=<HL<sub>1</sub><sup>(5)</sup>>];
    HL21 [label=<HL<sub>2</sub><sup>(1)</sup>>];
    HL22 [label=<HL<sub>2</sub><sup>(2)</sup>>];
    HL23 [label=<HL<sub>2</sub><sup>(3)</sup>>];
    HL24 [label=<HL<sub>2</sub><sup>(4)</sup>>];
    HL25 [label=<HL<sub>2</sub><sup>(5)</sup>>];
}
{
    node [shape=circle, color=yellow, style=filled, fillcolor=yellow];
    IPO [label=<IP<sub>O</sub>>];
    }
    {
    node [shape=circle, color=coral1, style=filled, fillcolor=coral1];
    CPO [label=<CP<sub>O</sub>>];
    }
    {
    node [shape=circle, color=chartreuse, style=filled, fillcolor=chartreuse];
    LWO1 [label=<LW<sub>O</sub><sup>(1)</sup>>];
    LWO2 [label=<LW<sub>O</sub><sup>(2)</sup>>];
    LWO3 [label=<LW<sub>O</sub><sup>(3)</sup>>];
}
    {
        rank=same;
        IPI1->IPI2->IPI3->CPI1->CPI2->CPI3->CPI4->LWI1->LWI2->LWI3;
    }
    {
        rank=same;
        HL11->HL12->HL13->HL14->HL15;
    }
    {
        rank=same;
        HL21->HL22->HL23->HL24->HL25;
    }
    {
        rank=same;
        IPO->CPO->LWO1->LWO2->LWO3;
    }
    l0 [shape=plaintext, label="layer 1 (input layer)"];
    l0->IPI1;
    {rank=same; l0;IPI1};
    l1 [shape=plaintext, label="layer 2 (hidden layer)"];
    l1->HL11;
    {rank=same; l1;HL11};
    l2 [shape=plaintext, label="layer 3 (hidden layer)"];
    l2->HL21;
    {rank=same; l2;HL21};
    l3 [shape=plaintext, label="layer 4 (output layer)"];
    l3->IPO;
    {rank=same; l3;IPO};
    edge[style=solid, tailport=e, headport=w];
    {IPI1;IPI2;IPI3;CPI1;CPI2;CPI3;CPI4;LWI1;LWI2;LWI3} -> {HL11;HL12;HL13;HL14;HL15};
    {HL11;HL12;HL13;HL14;HL15} -> {HL21;HL22;HL23;HL24;HL25};
    {HL21;HL22;HL23;HL24;HL25} -> {IPO;CPO;LWO1;LWO2;LWO3};
}
"""
s = Source(temp, filename="nn_schematic.gv", format="png")
=======
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:08:55 2020

@author: benso
"""

from graphviz import Source
temp = """
digraph G {
    rankdir = LR;
    splines=false;
    edge[style=invis];
    ranksep= 1.4;
    {
    node [shape=circle, color=yellow, style=filled, fillcolor=yellow]; #IP Input
    IPI1 [label=<cos(theta)>];
    IPI2 [label=<sin(theta)>]; 
    IPI3 [label=<Theta dot>];
    }
    {
    node [shape=circle, color=coral1, style=filled, fillcolor=coral1]; #CP Input
    CPI1 [label=<     X     >];
    CPI2 [label=<   X dot  >]; 
    CPI3 [label=<   Theta   >];
    CPI4 [label=<Theta dot>];
    }
    {
    node [shape=circle, color=chartreuse, style=filled, fillcolor=chartreuse]; #LW Input
    LWI1 [label=<Theta>];
    LWI2 [label=<Omega>]; 
    LWI3 [label=<LW<Footstate>];
}
{
    node [shape=circle, color=dodgerblue, style=filled, fillcolor=dodgerblue]; #HL1
    HL11 [label=<HL<sub>1</sub><sup>(1)</sup>>];
    HL12 [label=<HL<sub>1</sub><sup>(2)</sup>>];
    HL13 [label=<HL<sub>1</sub><sup>(3)</sup>>];
    HL14 [label=<HL<sub>1</sub><sup>(4)</sup>>];
    HL15 [label=<HL<sub>1</sub><sup>(5)</sup>>];
    HL21 [label=<HL<sub>2</sub><sup>(1)</sup>>];
    HL22 [label=<HL<sub>2</sub><sup>(2)</sup>>];
    HL23 [label=<HL<sub>2</sub><sup>(3)</sup>>];
    HL24 [label=<HL<sub>2</sub><sup>(4)</sup>>];
    HL25 [label=<HL<sub>2</sub><sup>(5)</sup>>];
}
{
    node [shape=circle, color=yellow, style=filled, fillcolor=yellow];
    IPO [label=<IP<sub>O</sub>>];
    }
    {
    node [shape=circle, color=coral1, style=filled, fillcolor=coral1];
    CPO [label=<CP<sub>O</sub>>];
    }
    {
    node [shape=circle, color=chartreuse, style=filled, fillcolor=chartreuse];
    LWO1 [label=<LW<sub>O</sub><sup>(1)</sup>>];
    LWO2 [label=<LW<sub>O</sub><sup>(2)</sup>>];
    LWO3 [label=<LW<sub>O</sub><sup>(3)</sup>>];
}
    {
        rank=same;
        IPI1->IPI2->IPI3->CPI1->CPI2->CPI3->CPI4->LWI1->LWI2->LWI3;
    }
    {
        rank=same;
        HL11->HL12->HL13->HL14->HL15;
    }
    {
        rank=same;
        HL21->HL22->HL23->HL24->HL25;
    }
    {
        rank=same;
        IPO->CPO->LWO1->LWO2->LWO3;
    }
    l0 [shape=plaintext, label="layer 1 (input layer)"];
    l0->IPI1;
    {rank=same; l0;IPI1};
    l1 [shape=plaintext, label="layer 2 (hidden layer)"];
    l1->HL11;
    {rank=same; l1;HL11};
    l2 [shape=plaintext, label="layer 3 (hidden layer)"];
    l2->HL21;
    {rank=same; l2;HL21};
    l3 [shape=plaintext, label="layer 4 (output layer)"];
    l3->IPO;
    {rank=same; l3;IPO};
    edge[style=solid, tailport=e, headport=w];
    {IPI1;IPI2;IPI3;CPI1;CPI2;CPI3;CPI4;LWI1;LWI2;LWI3} -> {HL11;HL12;HL13;HL14;HL15};
    {HL11;HL12;HL13;HL14;HL15} -> {HL21;HL22;HL23;HL24;HL25};
    {HL21;HL22;HL23;HL24;HL25} -> {IPO;CPO;LWO1;LWO2;LWO3};
}
"""
s = Source(temp, filename="nn_schematic.gv", format="png")
>>>>>>> e8fbe70b727e80f4fd5102d1e97bc3356029035b
s.view(quiet=True)