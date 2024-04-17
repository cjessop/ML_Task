from copy import deepcopy, copy
from collections.abc import Sequence, Mapping
import pathlib
from _items import Item, Call, Ref
from BaseMLClasses import PipelineLoader
import networkx

class pipeline():
    
    @classmethod
    def read(cls, filename):
        config = PipelineLoader.load_pipeline_yaml(filename)
        return cls(config)
    
    def __init__(self, configuration):
        self.config = deepcopy(configuration)
        self.params = self.config.pop('parameters', {})
        self.table_config = self.config.pop('tables', {})
        #default_table = Call(Table)
        self.state = copy(self.params)
        self.dag = networkx.DiGraph()
        contxt = {}
        items = {}

        for job, settings in self.config.items():
            self.dag.add_node(job, skip=False)

            if isinstance(settings, Item):
                items[job] = settings


    def execute(self, parameters={}):
        self.parameters=parameters
        self.state=copy(self.parameters)

    def write(self, filename, overwrite=False):
        suffix = pathlib.Path(filename).suffix.lower()