import igraph
import matplotlib.pyplot as plt
import numpy as np


class tree_results_fit:
    """
    The initialization of this class gathers all options used for calculation of FWI on CMIP6 data.
    'func_prepare_files' is then used for preparing all the files that will be used, such as the runs to use, the exceptions, etc.

    NB: Used during calculation of the FWI from CMIP6 data.
    """

    # --------------------
    # INITIALIZATION
    # --------------------
    def __init__(self, results_fit, do_empty_nodes=True, layout="rt_circular"):
        self.results_fit = results_fit
        self.do_empty_nodes = do_empty_nodes
        self.distrib2params = {'gaussian':['loc', 'scale'], 'GEV':['loc', 'scale', 'shape'], 'poisson':['loc', 'mu'], 'GPD':['loc', 'scale', 'shape'], 'skewnorm':['loc', 'scale', 'shape_skewnorm']}
        self.written_param = {'loc':'\\mu', 'scale':'\\sigma', 'shape':'\\xi', 'mu':'\\lambda', 'shape_skewnorm':'\\alpha'}
        self.written_distrib = {'GEV':'GEV', 'poisson':'Poisson', 'gaussian':'\mathcal{N}', 'GPD':'GPD', 'skewnorm':'S\mathcal{N}'}
        self.written_evol = {'constant':'cst', 'linear':'\Delta T', 'power2':'\Delta T^2', 'poly2':'\Delta T, \Delta T^2', 'power3':'\Delta T^3', 'poly3':'\Delta T, \Delta T^2, \Delta T^3'}
        self.round_acc = 2
        self.layout = layout
        
        # preparation
        self.prep_results()
        self.prep_colors()

        # positions before plot
        self.calculate_positions_nodes(layout=self.layout) # 'kk', 'auto', 'rt_circular', 'fr', 'rt'
    # --------------------
    # --------------------

    # --------------------
    # PREPARATION
    # --------------------
    def prep_results(self):
        # best fit:
        self.best_fit = self.results_fit[np.argmin( [fit[3] for fit in self.results_fit] )][0]
        
        # preparing items
        self.list_distrib = []
        self.dico_results = {}
        for i, res in enumerate( self.results_fit ):
            # identifying fit
            res_split = res[0].split('_')
            if res_split[0] not in self.list_distrib:
                self.list_distrib.append( res_split[0] )
                self.dico_results[ res_split[0] ] = {}
            # location / mu
            if res_split[1] not in self.dico_results[res_split[0]]:
                self.dico_results[res_split[0]][ res_split[1] ] = {}
            # scale
            if res_split[2] not in self.dico_results[res_split[0]][res_split[1]]:
                self.dico_results[res_split[0]][res_split[1]][ res_split[2] ] = {}
            # shape
            if len( self.distrib2params[res_split[0]] ) == 3:
                if res_split[3] not in self.dico_results[res_split[0]][res_split[1]][res_split[2]]:
                    self.dico_results[res_split[0]][res_split[1]][res_split[2]][res_split[3]] = {'value':np.round( res[-1], self.round_acc)}
            else:
                self.dico_results[res_split[0]][res_split[1]][res_split[2]]['value'] = np.round( res[-1], self.round_acc)
                
        if self.do_empty_nodes:
            # edit for readibility: will add empty nodes for distributions with 2 parameters
            for d in self.list_distrib:
                if len(self.distrib2params[d]) == 2:
                    new = {}
                    for opt1 in self.dico_results[d]:
                        new[opt1] = {}
                        for opt2 in self.dico_results[d][opt1]:
                            new[opt1][opt2] = {'empty_val':self.dico_results[d][opt1][opt2], 'empty':'empty'}
                    self.dico_results[d] = new
                
                
    def prep_colors(self):
        vals = [res[-1] for res in self.results_fit]
        self.min_vals, self.max_vals = np.min(vals), np.max(vals)
        
    def rgb(self, value, r_range=[0,255*0.8], g_range=[255*0.8,0], b_range=[0,0], a_range=[0.8,0.8]):
        ratio = np.min([np.max([0, (value-self.min_vals) / (self.max_vals - self.min_vals)]), 1])
        r = r_range[0] + ratio * (r_range[1] - r_range[0])
        g = g_range[0] + ratio * (g_range[1] - g_range[0])
        b = b_range[0] + ratio * (b_range[1] - b_range[0])
        norm = r + g + b
        a = a_range[0] + ratio * (a_range[1] - a_range[0])
        return int(255*r/norm), int(255*g/norm), int(255*b/norm), a

    def write_expression( self, param, evol ):
        if True:
            if evol in ['', 'empty', 'empty_val']:
                return evol
            else:
                return '$' + param + ': ' + self.written_evol[evol] + '$'
        else:
            if evol == 'constant':
                return '$'+param+'_0$'
            elif evol == 'linear':
                return '$'+param+'_0 + '+param+'_1 \Delta T_t$'
            elif evol == 'power2':
                return '$'+param+'_0 + '+param+'_1 \Delta T_t^2$'
            elif evol == 'poly2':
                return '$'+param+'_0 + '+param+'_1 \Delta T_t + '+param+'_2 \Delta T_t^2$'
            elif evol == 'power3':
                return '$'+param+'_0 + '+param+'_1 \Delta T_t^3$'
            elif evol == 'poly3':
                return '$'+param+'_0 + '+param+'_1 \Delta T_t + '+param+'_2 \Delta T_t^2$ + ' + param+'_3 \Delta T_t^3$'

    # --------------------
    # --------------------

    # --------------------
    # POSITIONS OF NODES
    # --------------------
    @staticmethod
    def find_angle(x, y):
        angle = np.arccos(x / np.sqrt(x**2 + y**2))
        if y < 0:
            angle *= -1
        return angle

    def calculate_positions_nodes(
        self,
        layout,
        **args
    ):
        # sorting best layouts: 'kk', 'auto', 'rt_circular', 'fr', 'rt'
        # additional layouts, but that hard to read: 'circle', 'dh', 'drl', 'drl_3d', 'grid', 'grid_3d', 'graphopt', 'fr_3d', 'kk_3d', 'lgl', 'mds', 'random', 'random_3d', 'sphere', 'star', 'sugiyama'
        # additional layouts, but that raises an issue: 'bipartite'

        # preparing tree
        self.labels = ["<b>Configuration</b>"]  # central node
        self.list_links = []  # tells which nodes are linked
        self.dico_mem, counter = {"Configuration": 0}, 0  # used to identify nodes
        self.is_to_color = [False] # tells which node will be colored
        self.color_to_do = [''] # tells which color for the node
        self.is_best_fit = [False]

        # preparing first level: distribution
        for i_d, d in enumerate(self.list_distrib):
            counter += 1
            self.dico_mem[d] = counter
            self.list_links.append((0, self.dico_mem[d]))
            self.is_best_fit.append( False )
            self.is_to_color.append( False )
            self.color_to_do.append( '' )
            #self.labels.append("<br>"+ str(d))
            self.labels.append("$\mathbf{"+ str(self.written_distrib[d])+"}$")

        # preparing second level: location / mu
        self.expressions = [] # listing them to ease changes in labels
        for i_d, d in enumerate(self.list_distrib):
            for i_opt, opt in enumerate(self.dico_results[d].keys()):
                counter += 1
                self.dico_mem[d + "_" + opt] = counter
                self.list_links.append(
                    (self.dico_mem[d], self.dico_mem[d + "_" + opt])
                )
                self.labels.append( self.write_expression(param=self.written_param[self.distrib2params[d][0]], evol=opt) )
                if self.labels[-1] not in self.expressions:
                    self.expressions.append( self.labels[-1] )
                self.is_best_fit.append( False )
                self.is_to_color.append( False )
                self.color_to_do.append( '' )

        # preparing third level: scale
        for i_d, d in enumerate(self.list_distrib):
            for i_opt1, opt1 in enumerate(self.dico_results[d].keys()):
                for i_opt2, opt2 in enumerate(self.dico_results[d][opt1].keys()):
                    counter += 1
                    self.dico_mem[d + "_" + opt1 + "_" + opt2] = counter
                    self.list_links.append(
                        (self.dico_mem[d + "_" + opt1], self.dico_mem[d + "_" + opt1 + "_" + opt2])
                    )
                    self.labels.append( self.write_expression(param=self.written_param[self.distrib2params[d][1]], evol=opt2) )
                    if self.labels[-1] not in self.expressions:
                        self.expressions.append( self.labels[-1] )
                    self.is_best_fit.append( False )
                    self.is_to_color.append( False )
                    self.color_to_do.append( '' )
                    
        # preparing fourth level: shape  OR  value for distrib with 2 params
        for i_d, d in enumerate(self.list_distrib):
            for i_opt1, opt1 in enumerate(self.dico_results[d].keys()):
                for i_opt2, opt2 in enumerate(self.dico_results[d][opt1].keys()):
                    if list(self.dico_results[d][opt1][opt2].keys()) != ['value']:
                        for i_opt3, opt3 in enumerate(self.dico_results[d][opt1][opt2].keys()):
                            counter += 1
                            self.dico_mem[d + "_" + opt1 + "_" + opt2 + "_" + opt3] = counter
                            self.list_links.append(
                                (self.dico_mem[d + "_" + opt1 + "_" + opt2], self.dico_mem[d + "_" + opt1 + "_" + opt2 + "_" + opt3])
                            )
                            if opt3 in ['', 'empty', 'empty_val']:
                                self.labels.append( self.write_expression(param='', evol=opt3) )
                            else:
                                self.labels.append( self.write_expression(param=self.written_param[self.distrib2params[d][2]], evol=opt3) )
                            if self.labels[-1] not in self.expressions:
                                self.expressions.append( self.labels[-1] )
                            self.is_best_fit.append( False )
                            self.is_to_color.append( False )
                            self.color_to_do.append( '' )
                    else:
                        # if here, means that this is a distribution with only 2 parameters.
                        counter += 1
                        self.dico_mem[d + "_" + opt1 + "_" + opt2 + "_" + "value"] = counter
                        self.list_links.append(
                            (self.dico_mem[d + "_" + opt1 + "_" + opt2], self.dico_mem[d + "_" + opt1 + "_" + opt2 + "_" + "value"])
                        )
                        self.labels.append("<b>" + str(self.dico_results[d][opt1][opt2]['value']) + "</b>")
                        self.is_to_color.append( True )
                        self.color_to_do.append( self.rgb(self.dico_results[d][opt1][opt2]['value']) )
                        self.is_best_fit.append( d + "_" + opt1 + "_" + opt2 == self.best_fit )
                        
        # preparing fifth level: value for distrib with 3 params
        for i_d, d in enumerate(self.list_distrib):
            for i_opt1, opt1 in enumerate(self.dico_results[d].keys()):
                for i_opt2, opt2 in enumerate(self.dico_results[d][opt1].keys()):
                    for i_opt3, opt3 in enumerate(self.dico_results[d][opt1][opt2].keys()):
                        if opt3 != "value":
                            counter += 1
                            self.dico_mem[d + "_" + opt1 + "_" + opt2 + "_" + opt3 + "_" + "value"] = counter
                            self.list_links.append(
                                (self.dico_mem[d + "_" + opt1 + "_" + opt2 + "_" + opt3], self.dico_mem[d + "_" + opt1 + "_" + opt2 + "_" + opt3 + "_" + "value"])
                            )
                            if opt3 == 'empty':
                                self.labels.append("")
                                self.is_to_color.append( False )
                                self.color_to_do.append( '' )
                                self.is_best_fit.append( d + "_" + opt1 + "_" + opt2 == self.best_fit )
                            else:# opt3 == 'empty_val' or a shape.
                                self.labels.append("<b>" + str(self.dico_results[d][opt1][opt2][opt3]['value']) + "</b>")
                                self.is_to_color.append( True )
                                self.color_to_do.append( self.rgb(self.dico_results[d][opt1][opt2][opt3]['value']) )
                                if opt3 == 'empty_val':
                                    self.is_best_fit.append( d + "_" + opt1 + "_" + opt2 == self.best_fit )
                                else:
                                    self.is_best_fit.append( d + "_" + opt1 + "_" + opt2 + "_" + opt3 == self.best_fit )

        # dealing with layout
        nr_vertices = len(self.list_links) + 1
        G = igraph.Graph(self.list_links)
        self.lay = G.layout(layout, **args)  # lots of layouts to try

        if True:#layout in ["rt_circular"]:
            # edits
            XX = [self.lay[k][0] for k in range(len(self.lay))]
            YY = [self.lay[k][1] for k in range(len(self.lay))]
            # saving edited layout
            self.lay = [[XX[k], YY[k]] for k in range(len(XX))]

        # positions
        positions = {k: self.lay[k] for k in range(nr_vertices)}
        # self.Msens = max( [self.lay[k][1] for k in range(nr_vertices)] )

        # edges
        es = igraph.EdgeSeq(G)  # sequence of edges
        E = [e.tuple for e in G.es]  # list of edges

        # final preparation
        self.Xn = [positions[k][0] for k in range(len(positions))]
        # self.Yn = [2*self.Msens-positions[k][1] for k in range(len(positions))]
        self.Yn = [positions[k][1] for k in range(len(positions))]
        self.Xe, self.Ye = [], []
        for edge in E:
            self.Xe += [positions[edge[0]][0], positions[edge[1]][0], None]
            # self.Ye+=[2*self.Msens-positions[ edge[0] ][1], 2*self.Msens-positions[ edge[1] ][1], None]
            self.Ye += [positions[edge[0]][1], positions[edge[1]][1], None]

    # --------------------
    # --------------------

    
    # --------------------
    # ANNOTATIONS
    # --------------------
    def make_annotations(self, font_size, font_color="rgb(0,0,0)"):
        # central node
        annotations = [
            dict(
                text="<b>Configuration</b>",
                x=self.Xn[0],
                y=self.Yn[0],
                xref="x",
                yref="y",
                font=dict(color=font_color, size=font_size["configuration"]),
                textangle=0,
                showarrow=False,
            )
        ]

        # nodes with links
        for k in range(len(self.list_links)):
            # start & end of the link
            start, end = self.list_links[k]

            if self.labels[end] in ['', 'empty', 'empty_val']:
                pass
            
            else:
                # properties
                if self.labels[end] == "<b>Configuration</b>":
                    ftsz = font_size["configuration"]
                elif self.labels[end] in ["$\mathbf{"+ str(self.written_distrib[d])+"}$" for d in self.list_distrib]:
                    ftsz = font_size["distribution"]
                elif self.labels[end] in self.expressions:
                    ftsz = font_size["expression"]
                else:
                    ftsz = font_size["BIC"]
                # rotation of the text
                ang = (
                    self.find_angle(
                        x=self.Xn[end] - self.Xn[start], y=self.Yn[end] - self.Yn[start]
                    )
                    * 360
                    / (2 * np.pi)
                )
                # crazy angles in update_layout... (>_<)!
                if (-90 <= ang) and (ang < 90):
                    ang = -ang
                else:
                    ang = 180 - ang

                ftcl = font_color
                annotations.append(
                    dict(
                        text=self.labels[end],
                        x=self.Xn[end],
                        y=self.Yn[end],
                        xref="x",
                        yref="y",
                        font=dict(color=ftcl, size=ftsz),
                        textangle=ang,
                        showarrow=False,
                    )
                )
        return annotations

    # rotation_txt!
    # --------------------
    # --------------------

    # --------------------
    # PLOT
    # --------------------
    def plot(
        self,
        figsize=(1000, 1000),
        colors={ "lines": "rgb(200,200,200)", "nodes": "rgb(100,100,100)", "edges": "rgb(100,100,100)", "background": "rgb(248,248,248)", "text": "rgb(0,0,0)" },
        sizes={"dots": 25, "configuration": 15, "distribution": 15, "expression": 8, "BIC": 8, 'empty':1},
    ):
        """
        Plot the tree

        args:
            figsize: tuple, list or numpy.array
                (width,height)

            colors: dict
                Colors for lines, nodes, edges, background, text

            sizes: dict
                Sizes for dots & other elts
        """
        # ploting
        fig = go.Figure()
        # looping on nodes to make sure to have empty nodes
        for i in np.arange( len(self.list_links) ):
            start, end = self.list_links[i]
            # properties
            if self.labels[end] in ['', 'empty']:
                col = colors["background"]
                lw = 0
            else:
                col = colors["lines"]
                lw = 2
            # plot line
            fig.add_trace(
                go.Scatter(
                    x=self.Xe[3*i:3*(i+1)],
                    y=self.Ye[3*i:3*(i+1)],
                    mode="lines",
                    line=dict(color=col, width=lw),
                    hoverinfo="none",
                )
            )
        # looping on nodes to make sure to have color
        for i in np.arange( len(self.Xn) ):
            if self.is_to_color[i]:
                ftcl = "rgb("+ ','.join([str(c) for c in self.color_to_do[i]]) +")"
            else:
                ftcl = colors["nodes"]
            if self.labels[i] in ['', 'empty', 'empty_val']:
                sz = sizes["empty"]
                col = colors["background"]
                ftcl = colors["background"]
            else:
                sz = sizes["dots"]
                col = colors["edges"]
            if (self.is_best_fit[i]) and (self.labels[i] not in ['', 'empty']):
                lw = 8
                col = 'green'
            else:
                lw = 1
            # plot node
            fig.add_trace(
                go.Scatter(x=[self.Xn[i]], y=[self.Yn[i]], mode="markers",
                           marker=dict(symbol="circle", size=sz, color=ftcl, line=dict(color=col, width=lw)), text=[self.labels[i]], hoverinfo="text", opacity=0.8 )
            )

        # Actualizing plot
        axis = dict(
            showline=False,  # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
        )

        self.annotations = self.make_annotations(
            font_size=sizes, font_color=colors["text"]
        )
        fig.update_layout(
            annotations=self.annotations,
            font_size=10,  # not required, hiding this one
            showlegend=False,
            xaxis=axis,
            yaxis=axis,
            width=figsize[0],
            height=figsize[1],
            margin=dict(l=40, r=40, b=40, t=40),
            hovermode="closest",
            plot_bgcolor=colors["background"],
        )
        fig.write_image( name_save + '.png' )
        fig.write_image( name_save + '.pdf' )
        plt.close(fig)
        return fig

    # --------------------
    # --------------------