import pickle
import torch
import numpy
import copy


class BasicData:
    def __init__(self):
        self.data = None
        
    def to_device(device):
        pass

    def save(self, file_path):
        self.to_device("cpu")
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
        
    def copy(self):
        return copy.deepcopy(self)
    
##############################################################################################################
##############################################################################################################


#
# scRNA-seq data class
#

# We build a data class to keep our data in single cell RNA sequencing analysis
class SCData(BasicData):
    """
    class for handling data matrices in single cell RNA sequencing analysis
    """

    def __init__(self, expression, cell, gene, location=None):
        # gene expression data
        self.raw = RawData(expression, cell, gene, location)   # raw data
        self.svg = SelectVariableGeneData()   # selected variable genes data
#         self.fae = FeatureAEData()  # feature auto-encoder data
        self.gae = GraphAEData()  # graph auto-encoder data
        self.traj = Trajectory()

        # information data
        
        self.raw_graph = GraphTopology()  # graph information # graph before diffusion if not None
        self.dif_graph = GraphTopology()  # graph information # diffusion used graph if not None
        self.graph = GraphTopology()  # graph information # follow-up used graph
        self.graph_le = GraphTopology()  # local equilibrium graph information # follow-up used graph
        self.graph_ge = GraphTopology()  # global equilibrium graph information # follow-up used graph
        
        
        self.cluster = None  # cluster information
        self.cluster_names = None  # cluster information
        self.marker = None  # markers information
        self.dlog = SCRNASeqDataLog()  #  other data related information
        self.record = SCRNASeqDataRecord()
        
    def model_state(self):
        scmodel = SCModelState()
#         scmodel.fae.model = self.fae.model
#         scmodel.fae.args = self.fae.args
        scmodel.gae.model = self.gae.model
        scmodel.gae.args = self.gae.args
        
        return scmodel
    
    def remove_model(self):
        self.gae.model = None
        self.gae.args = None
        
    def load_gnd_edge_attention(self, attention_type='attention'):
        for i in range(len(self.gae.gnd_graph)):
            self.gae.gnd_graph[i].load_edge_attention(attention_type=attention_type)

    def to_torch(self, dtype=torch.float32):  # transfer data type to PyTorch tensor
        self.raw.to_torch(dtype=dtype)
        self.svg.to_torch(dtype=dtype)
        self.gae.to_torch(dtype=dtype)
        self.raw_graph.to_torch(dtype=torch.int64, dtype_attention=dtype)
        self.dif_graph.to_torch(dtype=torch.int64, dtype_attention=dtype)
        self.graph.to_torch(dtype=torch.int64, dtype_attention=dtype)
        self.graph_le.to_torch(dtype=torch.int64, dtype_attention=dtype) 
        self.graph_ge.to_torch(dtype=torch.int64, dtype_attention=dtype)

    def to_numpy(self):  # transfer data type to NumPy array
        self.raw.to_numpy()
        self.svg.to_numpy()
        self.gae.to_numpy()
        self.raw_graph.to_numpy()
        self.dif_graph.to_numpy()
        self.graph.to_numpy()
        self.graph_le.to_numpy()
        self.graph_ge.to_numpy()

    def to_device(self, device):  # load data to device
        self.raw.to_device(device)
        self.svg.to_device(device)
        self.gae.to_device(device)
        self.raw_graph.to_device(device)
        self.dif_graph.to_device(device)
        self.graph.to_device(device)
        self.graph_le.to_device(device)
        self.graph_ge.to_device(device)
    
    # print() will call this function
    def __str__(self):
        return f"SCData object. Use SCData.print_data() method to check the data structure."
    
    def print_data(self, show_all = False):
        
        self.raw.print_data(show_all=show_all)
        self.svg.print_data(show_all=show_all)
        self.gae.print_data(show_all=show_all)
        
        print("  ")
        
        self.raw_graph.print_data(graph_name="raw_graph")
        self.dif_graph.print_data(graph_name="dif_graph")
        self.graph.print_data(graph_name="graph")
        self.graph_le.print_data(graph_name="graph_le")
        self.graph_ge.print_data(graph_name="graph_ge")
        self.traj.print_data()
        
        print("SCData.cluster: cluster information.")
        print("------None") if self.cluster is None else print(f"------len: {len(self.cluster)}", self.cluster)
        print("SCData.cluster_names: cluster information.")
        print("------None") if self.cluster_names is None else print(f"------len: {len(self.cluster_names)}", self.cluster_names)
        print("SCData.marker: marker information.")
        print("------None") if self.marker is None else print(f"------len: {len(self.marker)}", self.marker)
        
        self.dlog.print_data(show_all=show_all)
        
#     def copy(self):
#         return copy.deepcopy(self)
        


# Helpful data classes to class SCRNASeqData

class RawData(BasicData):
    """
    Class to keep raw single cell data class
    """
    def __init__(self, expression, cell, gene, location):
        self.expr = expression
        self.log = None
        self.cell = cell
        self.gene = gene
        self.lct = location

    def to_torch(self, dtype=torch.float32):  # transfer data type to PyTorch tensor
        self.expr = torch_load(self.expr, dtype=dtype)
        self.log = torch_load(self.log, dtype=dtype)
        self.lct = torch_load(self.lct, dtype=dtype)

    def to_numpy(self):  # transfer data type to NumPy array
        self.expr = numpy_load(self.expr)
        self.log = numpy_load(self.log)
        self.lct = numpy_load(self.lct)

    def to_device(self, device):  # load data to device
        self.expr = move_to_device(self.expr, device)
        self.log = move_to_device(self.log, device)
        self.lct = move_to_device(self.lct, device)
        
    def print_data(self, show_all=True):
        print("SCData.raw: Preprossed raw data (loaded raw data if no preprocessing).")
        print("------raw.expr: " + describe_string(self.expr))
        print("------raw.log: " + describe_string(self.log))
        print("------raw.cell: " + describe_string(self.cell))
        print("------raw.gene: " + describe_string(self.gene))
        print("------raw.lct: " + describe_string(self.lct))
        


class SelectVariableGeneData(BasicData):
    """
    Class to keep selected variable genes data
    """
    def __init__(self):
        self.expr = None
        self.log = None
        self.gene = None

    def to_torch(self, dtype=torch.float32):  # transfer data type to PyTorch tensor
        self.expr = torch_load(self.expr, dtype=dtype)
        self.log = torch_load(self.log, dtype=dtype)

    def to_numpy(self):  # transfer data type to NumPy array
        self.expr = numpy_load(self.expr)
        self.log = numpy_load(self.log)

    def to_device(self, device):  # load data to device
        self.expr = move_to_device(self.expr, device)
        self.log = move_to_device(self.log, device)
        
    def print_data(self, show_all=True):
        print("SCData.svg: Selected variable genes data.")
        print("------svg.expr: " + describe_string(self.expr))
        print("------svg.log: " + describe_string(self.log))
        print("------svg.gene: " + describe_string(self.gene))
        

class GraphAEData(BasicData):
    """
    Class to keep graph auto-encoder data
    """
    def __init__(self):
        self.input = None
        self.output = None
        self.indata = None
        self.outdata = None
        self.umap_in = None
        self.umap_out = None
        self.model = None
        self.args = None
        self.attention = None
        self.gnd = None
        self.gnd_embed = None
        self.gnd_graph = None
        
        # Record graph-AE information.
        self.dlog = None

    def to_torch(self, dtype=torch.float32):  # transfer data type to PyTorch tensor
        self.input = torch_load(self.input, dtype=dtype)
        self.output = torch_load(self.output, dtype=dtype)
        self.indata = torch_load(self.indata, dtype=dtype)
        self.outdata = torch_load(self.outdata, dtype=dtype)
        self.umap_in = torch_load(self.umap_in, dtype=dtype)
        self.umap_out = torch_load(self.umap_out, dtype=dtype)
        if self.attention is not None:
            self.attention = torch_load(self.attention, dtype=dtype, components_only=True)
        if self.gnd is not None:
            for i in range(len(self.gnd)):
                self.gnd[i] = torch_load(self.gnd[i], dtype=dtype, components_only=True)
        if self.gnd_embed is not None:
            for i in range(len(self.gnd_embed)):
                self.gnd_embed[i] = torch_load(self.gnd_embed[i], dtype=dtype, components_only=True)
        if self.gnd_graph is not None:
            for i in range(len(self.gnd_graph)):
                self.gnd_graph[i].to_torch(dtype=torch.int64, dtype_attention=dtype)

    def to_numpy(self):  # transfer data type to NumPy array
        self.input = numpy_load(self.input)
        self.output = numpy_load(self.output)
        self.indata = numpy_load(self.indata)
        self.outdata = numpy_load(self.outdata)
        self.umap_in = numpy_load(self.umap_in)
        self.umap_out = numpy_load(self.umap_out)
        if self.attention is not None:
            self.attention = numpy_load(self.attention, components_only=True)
        if self.gnd is not None:
            for i in range(len(self.gnd)):
                self.gnd[i] = numpy_load(self.gnd[i],components_only=True)
        if self.gnd_embed is not None:
            for i in range(len(self.gnd_embed)):
                self.gnd_embed[i] = numpy_load(self.gnd_embed[i],components_only=True)
        if self.gnd_graph is not None:
            for i in range(len(self.gnd_graph)):
                self.gnd_graph[i].to_numpy()

    def to_device(self, device):  # load data to device
        self.input = move_to_device(self.input, device)
        self.output = move_to_device(self.output, device)
        self.indata = move_to_device(self.indata, device)
        self.outdata = move_to_device(self.outdata, device)
        self.umap_in = move_to_device(self.umap_in, device)
        self.umap_out = move_to_device(self.umap_out, device)
        if self.model is not None:
            self.model = move_to_device(self.model, device)
        if self.attention is not None:
            self.attention = move_to_device(self.attention, device)
        if self.gnd is not None:
            for i in range(len(self.gnd)):
                self.gnd[i] = move_to_device(self.gnd[i], device)
        if self.gnd_embed is not None:
            for i in range(len(self.gnd_embed)):
                self.gnd_embed[i] = move_to_device(self.gnd_embed[i], device)
        if self.gnd_graph is not None:
            for i in range(len(self.gnd_graph)):
                self.gnd_graph[i].to_device(device)
        
    def print_data(self, show_all=True):
        print("SCData.gae: Graph auto-encoder data.")
        print("------gae.input: " + describe_string(self.input))
        print("------gae.output: " + describe_string(self.output))
        print("------gae.indata: " + describe_string(self.indata))
        print("------gae.outdata: " + describe_string(self.outdata))
        print("------gae.umap_in: " + describe_string(self.umap_in))
        print("------gae.umap_out: " + describe_string(self.umap_out))
        print("------gae.model: " + model_describe_string(self.model))
        print("------gae.args: ", self.args)
        
        # Attention print
        string = "None" if self.attention is None else describe_string(self.attention) + " Components: " + describe_string(self.attention[0])
        print("------gae.attention: " + string)
        # graph neural diffusion print
        print("------gae.gnd: " + describe_string(self.gnd))
        print("------gae.gnd_embed: " + describe_string(self.gnd_embed))
        print("------gae.gnd_graph: " + describe_string(self.gnd_graph))
        
#         string = "None" if self.gnd is None else describe_string(self.gnd) + " Components: " + describe_string(self.gnd[0])
#         print("------gae.gnd: " + string)



class GraphTopology(BasicData):
    """
    Class to keep graph information(topology, such as adjacency, edge index and edge list).
    """
    def __init__(self):
        self.adj = None
        self.edge_index = None
        self.edge_list = None
        self.weighted_edge_list = None
        self.edge_dict = None
        self.heads_attention = None
        self.attention = None
        self.adjusted_attention = None
        self.edge_distance = None  # torch tensor
        self.edge_distance_inverse = None  # torch tensor
    
    def load_edge_attention(self, attention_type='attention'):
        weight_list = list(numpy_load(self.attention)) if attention_type == "attention" else list(numpy_load(self.adjusted_attention))
        self.weighted_edge_list = [(edge[0], edge[1], weight) for edge, weight in zip(self.edge_list, weight_list)]
        

    def to_torch(self, dtype=torch.int64, dtype_attention=torch.float32):  # transfer data type to PyTorch tensor
        self.adj = torch_load(self.adj, dtype=dtype)
        self.edge_index = torch_load(self.edge_index, dtype=dtype)
        self.heads_attention = torch_load(self.heads_attention, dtype=dtype_attention)
        self.attention = torch_load(self.attention, dtype=dtype_attention)
        self.adjusted_attention = torch_load(self.adjusted_attention, dtype=dtype_attention)

    def to_numpy(self):  # transfer data type to NumPy array
        self.adj = numpy_load(self.adj)
        self.edge_index = numpy_load(self.edge_index)
        self.heads_attention = numpy_load(self.heads_attention)
        self.attention = numpy_load(self.attention)
        self.adjusted_attention = numpy_load(self.adjusted_attention)

    def to_device(self, device):  # load data to device
        self.adj = move_to_device(self.adj, device)
        self.edge_index = move_to_device(self.edge_index, device)
        self.heads_attention = move_to_device(self.heads_attention, device)
        self.attention = move_to_device(self.attention, device)
        self.adjusted_attention = move_to_device(self.adjusted_attention, device)
        
    def print_data(self, graph_name="graph", show_all=True):
        print("SCData."+graph_name+": Graph information.")
        print("------"+graph_name+".adj: " + describe_string(self.adj))
        print("------"+graph_name+".edge_index: " + describe_string(self.edge_index))
        print("------"+graph_name+".edge_list: " + describe_string(self.edge_list))
        print("------"+graph_name+".weighted_edge_list: " + describe_string(self.weighted_edge_list))
        print("------"+graph_name+".edge_dict: " + describe_string(self.edge_dict))
        print("------"+graph_name+".heads_attention: " + describe_string(self.heads_attention))
        print("------"+graph_name+".attention: " + describe_string(self.attention))
        print("------"+graph_name+".adjusted_attention: " + describe_string(self.adjusted_attention))
        
        
class Trajectory(GraphTopology):
    """
    Class to record data of trajectory
    """
    def __init__(self, k=None, data_type = "gae"):
        super().__init__()
        self.cluster = None
        self.nodes = None
        self.expr = None
        self.smlr_edge_list = None
        self.traj_edge_list = None
        
    def print_data(self, show_all=True):
        print("SCData.traj: Trajectory information.")
        print("------traj.expr: " + describe_string(self.expr))
        print("------traj.nodes: " + describe_string(self.nodes))
#         print("------traj.cluster: " + describe_string(self.cluster))
#         print("------traj.adj: " + describe_string(self.adj))
#         print("------traj.edge_index: " + describe_string(self.edge_index))
        print("------traj.edge_list: " + describe_string(self.edge_list))
        print("------traj.self_edge_list: " + describe_string(self.self_edge_list))
        print("------traj.norm_edge_list: " + describe_string(self.norm_edge_list))
#         print("------traj.edge_dict: " + describe_string(self.edge_dict))
#         print("------traj.knn_indices: " + describe_string(self.knn_indices))
        
    

class SCRNASeqDataLog(BasicData):
    """
    Class to record data related information
    """
    def __init__(self):
        self.load = None
        self.raw = None
        self.svg = None
        self.fae = None
        self.gae = None
        self.cluster = None
        self.traj = None
        
    def print_data(self, show_all=True):
        print("SCData.dlog: Recorded information.")
        print("------dlog.load: ", self.load)
        print("------dlog.raw: ", self.raw)
        print("------dlog.svg: ", self.svg)
        print("------dlog.fae: ", self.fae)
        print("------dlog.gae: ", self.gae)
        print("------dlog.cluster: ", self.gae)
        print("------dlog.traj: ", self.traj)
        
class SCRNASeqDataRecord(BasicData):
    """
    Class to record data related information
    """
    def __init__(self):
        self.load = None
        self.raw = None
        self.svg = None
        self.fae = None
        self.gae = None
        self.cluster = None
        self.traj = None
        
    def print_data(self, show_all=True):
        print("------record.cluster: " + describe_string(self.cluster))
        print("------record.traj: " + describe_string(self.traj))


# Helpful data classes to class SCRNASeqData end

# Helpful functions

def torch_load(data, dtype=torch.float32, components_only = False):
    """
    Transfer data type from NumPy array or list to PyTorch tensor
    """
    if data is None:
        return torch.tensor([], dtype=dtype)
    elif isinstance(data, torch.Tensor):
        return data if data.dtype == dtype else data.to(dtype)
    elif isinstance(data, numpy.ndarray):
        return torch.tensor(data, dtype=dtype)
    elif isinstance(data, list):
        if components_only == False:
            return torch.tensor(data, dtype=dtype)
        else:
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i] if data[i].dtype == dtype else data[i].to(dtype)
                elif isinstance(data[i], numpy.ndarray):
                    data[i] = torch.tensor(data[i], dtype=dtype)
            return data
    elif isinstance(data, dict):
        if components_only == False:
            raise TypeError("Load dictionary to torch.tensor requires argument: components_only=True.")
        else:
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key] if data[key].dtype == dtype else data[key].to(dtype)
                elif isinstance(data[key], numpy.ndarray):
                    data[key] = torch.tensor(data[key], dtype=dtype)
            return data
    else:
        raise TypeError("Unsupported data type. Expected a PyTorch tensor, a NumPy array, or a list.")

def numpy_load(data,components_only = False, ):
    """
    Transfer data type from PyTorch tensor or list to NumPy array
    """
    if data is None:
        return numpy.array([])
    elif isinstance(data, numpy.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        if components_only == False:
            return numpy.array(data)
        else:
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].detach().cpu().numpy()
            return data
    elif isinstance(data, dict):
        if components_only == False:
            raise TypeError("Load dictionary to numpy.ndarray requires argument: components_only=True.")
        else:
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].detach().cpu().numpy()
            return data
    else:
        raise TypeError("Unsupported data type. Expected a NumPy array, a PyTorch tensor, or a list.")


def move_to_device(data, device):  
    """
    Load data to device.
    """
    if isinstance(data, torch.Tensor):  
        return data.to(device)
    
    elif isinstance(data, list):
        for i in range(len(data)):
            if isinstance(data[i], torch.Tensor):
                data[i] = data[i].to(device)
        return data
    
    elif isinstance(data, dict):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    
    else:
        return data

            
    
def describe_string(data):
    
    if data is None:
        return "None"
    else:
        if isinstance(data, numpy.ndarray):
            return f'numpy.narray, shape={data.shape}, dtype={data.dtype}.'
        elif isinstance(data, torch.Tensor):
            return f'torch.Tensor, size={data.size()}, dtype={data.dtype}, device={data.device}.'
        elif isinstance(data, list):
            return f'list, length={len(data)}.'
        elif isinstance(data, dict):
            return f'dict, length={len(data)}.'
        
def model_describe_string(data):
    
    if isinstance(data, dict):
        st0 = f'model_dict, length={len(data)}, ' + '{'
        
        for key, value in data.items():
            st0 = st0 + key + ': ' + describe_string(value) + ','
            
        st0 = st0 + '}.'
        
        return st0
         
    else:
        return "None"
                

##############################################################################################################
##############################################################################################################

#
# scRNA-seq model state class
#

# We build a model state class to keep our model states in single cell RNA sequencing analysis
class SCModelState(BasicData):
    """
    class for keeping model states in single cell RNA sequencing analysis
    """

    def __init__(self):
        self.fae = FeatureAEState()
        self.gae = GraphAEState()



# Helpful classes to class SCRNASeqModelState

class FeatureAEState(BasicData):
    def __init__(self):
        self.model = None
        self.args = None


class GraphAEState:
    def __init__(self):
        self.model = None
        self.args = None

# Helpful data classes to class SCRNASeqModelState end


