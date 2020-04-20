from graphviz import Source

temp = """

digraph G {
    rankdir = LR;
    splines=false;
    edge[style=invis];
    ranksep= 2.5;
    {
    node [shape=circle, fontsize = 29, fixedsize=true, width = 2.0, height = 2.0, color=yellow, style=filled, fillcolor=yellow];
    IPI1 [label=<cos(&#920;)>];
    IPI2 [label=<sin(&#920;)>]; 
    IPI3 [label=<&#969; >];
    }
    {
    node [shape=circle, fontsize = 29, fixedsize=true, width = 2.0, height = 2.0, color=chartreuse, style=filled, fillcolor=chartreuse];
    CPI1 [label=<      X      >];
    CPI2 [label=<   X dot  >]; 
    CPI3 [label=<   &#920;   >];
    CPI4 [label=<   &#969;   >];
    }
    {
    node [shape=circle, fontsize = 29, fixedsize=true, width = 2.0, height = 2.0, color=coral1, style=filled, fillcolor=coral1];
    LWI1 [label=<    &#920;   >];
    LWI2 [label=<   &#969;    >];
    LWI3 [label=<Foot State>];
    
}
{
    node [shape=circle, fontsize = 30, fixedsize=true, width = 2.0, height = 2.0, color=dodgerblue, style=filled, fillcolor=dodgerblue];
    a12 [label=<    1      >];
    a22 [label=<    2      >];
    a32 [label=<    3      >];
    a42 [label=<    4      >];
    a52 [label=<    5      >];
    a13 [label=<    1      >];
    a23 [label=<    2      >];
    a33 [label=<    3      >];
    a43 [label=<    4      >];
    a53 [label=<    5      >];
}
{
    node [shape=circle, fontsize = 25, fixedsize=true, width = 2.0, height = 2.0, color=yellow, style=filled, fillcolor=yellow];
    IPO [label=<L/R Force   >];
    }
    {
    node [shape=circle, fontsize = 25, fixedsize=true, width = 2.0, height = 2.0, color=chartreuse, style=filled, fillcolor=chartreuse];
    CPO [label=<L/R Velocity>]; 
    }
    {
    node [shape=circle, fontsize = 20, fixedsize=true, width = 2.0, height = 2.0, color=coral1, style=filled, fillcolor=coral1];
    LWO1 [label=<Forward Force>]; 
    LWO2 [label=<Backward Force>]; 
    LWO3 [label=<Leg Pos. Up/Down>]; 
}
    {
        rank=same;
        IPI1->IPI2->IPI3->CPI1->CPI2->CPI3->CPI4->LWI1->LWI2->LWI3;
    }
    {
        rank=same;
        a12->a22->a32->a42->a52;
    }
    {
        rank=same;
        a13->a23->a33->a43->a53;
    }
    {
        rank=same;
        IPO->CPO->LWO1->LWO2->LWO3;
    }
    l0 [shape=plaintext, label="layer 1 (input layer)"];
    l0->IPI1;
    {rank=same; l0;IPI1};
    l1 [shape=plaintext, label="layer 2 (hidden layer)"];
    l1->a12;
    {rank=same; l1;a12};
    l2 [shape=plaintext, label="layer 3 (hidden layer)"];
    l2->a13;
    {rank=same; l2;a13};
    l3 [shape=plaintext, label="layer 4 (output layer)"];
    l3->IPO;
    {rank=same; l3;IPO};
    edge[style=solid, tailport=e, headport=w];
    {IPI1;IPI2;IPI3;CPI1;CPI2;CPI3;CPI4;LWI1;LWI2;LWI3} -> {a12;a22;a32;a42;a52};
    {a12;a22;a32;a42;a52} -> {a13;a23;a33;a43;a53};
    {a13;a23;a33;a43;a53} -> {IPO;CPO;LWO1;LWO2;LWO3};
}
"""
s = Source(temp, filename="nn_schematic.gv", format="png")
s.view()