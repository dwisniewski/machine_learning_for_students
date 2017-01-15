from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from scipy import array, newaxis

# Marcin ---
import numpy as np
tf = None

from IPython.core.display import display, HTML
from IPython.display import clear_output, Image, display, HTML
# ---

def visualize_cost_function(sizes_transformed, prices_transformed, w0_values, w1_values, loss_function):
	points = []
	for i in w0_values:
	    for j in w1_values:
	        points.append([i, j, loss_function(sizes_transformed, prices_transformed, i,j)])
	points = array(points)

	Xs = points[:,0]
	Ys = points[:,1]
	Zs = points[:,2]


	fig = plt.figure()


	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=plt.cm.coolwarm, linewidth=0)

	fig.colorbar(surf)
	ax.xaxis.set_major_locator(MaxNLocator(5))
	ax.yaxis.set_major_locator(MaxNLocator(6))
	ax.zaxis.set_major_locator(MaxNLocator(5))
	ax.set_title('Wartosc f.kosztu w dziedzinie wartosci w0 i w1')
	ax.set_zlabel('Wartosc f. kosztu')
	ax.set_xlabel('Wartosc wagi w0')
	ax.set_ylabel('Wartosc wagi w1')
	fig.tight_layout()

	plt.show() 
	
	

# Marcin ---
def execute_tf_graph(outputs, inputs=None):
    if tf is None:
    	import tensorflow as tf
    
    if type(outputs) not in {list, tuple}:
        outputs = [outputs]
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        ret = sess.run(outputs, feed_dict=inputs)
        
    return ret
    
    
def strip_consts(graph_def, max_const_size=32):
    strip_def = tf.GraphDef()
    
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def=None, width=600, height=300, max_const_size=32, ungroup_gradients=False):
    if not graph_def:
        graph_def = tf.get_default_graph().as_graph_def()
        
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    data = str(strip_def)
    if ungroup_gradients:
        data = data.replace('"gradients/', '"b_')
        #print(data)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(data), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:{}px;height:{}px;border:0" srcdoc="{}"></iframe>
    """.format(width, height, code.replace('"', '&quot;'))
    display(HTML(iframe))
# ---
