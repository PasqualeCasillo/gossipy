import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, RandomVerticalFlip
from gossipy.core import AntiEntropyProtocol, CreateModelMode, StaticP2PNetwork
from gossipy.data import DataDispatcher

from gossipy.model import TorchModel
from gossipy.data.handler import ClassificationDataHandler
from gossipy.model.handler import TorchModelHandler
from gossipy.node import PENSNode
from gossipy.simul import GossipSimulator, SimulationReport
from gossipy.data import get_CIFAR10
from gossipy.utils import plot_evaluation

# AUTHORSHIP
__version__ = "0.0.1"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2022, gossipy"
__license__ = "MIT"
__maintainer__ = "Mirko Polato, PhD"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#


class CIFAR10Net(TorchModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def init_weights(self, *args, **kwargs) -> None:
        def _init_weights(m: nn.Module):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        #self.apply(_init_weights)
        pass

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def __repr__(self) -> str:
        return "CIFAR10Net(size=%d)" %self.get_size()


class CustomDataDispatcher(DataDispatcher):
    def assign(self, seed: int = 42) -> None:
        self.tr_assignments = [[] for _ in range(self.n)]
        self.te_assignments = [[] for _ in range(self.n)]

        n_ex = self.data_handler.size()
        ex_x_user = math.ceil(n_ex / self.n)

        for idx, i in enumerate(range(0, n_ex, ex_x_user)):
            self.tr_assignments[idx] = list(range(i, min(i + ex_x_user, n_ex)))

        if self.eval_on_user:
            n_eval_ex = self.data_handler.eval_size()
            eval_ex_x_user = math.ceil(n_eval_ex / self.n)
            for idx, i in enumerate(range(0, n_eval_ex, eval_ex_x_user)):
                self.te_assignments[idx] = list(range(i, min(i + eval_ex_x_user, n_eval_ex)))

# Dataset loading
transform = Compose([Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
rotate180 = RandomVerticalFlip(p=1.0)
train_set, test_set = get_CIFAR10()
Xtr, ytr = (transform(train_set[0]), train_set[1])
Xte, yte = (transform(test_set[0]), test_set[1])
rotated_Xtr, rotated_Xte = rotate180(Xtr), rotate180(Xte)

half = train_set[0].size(0) // 2
train_set = torch.cat((Xtr[:half], rotated_Xtr[half:])), torch.cat((ytr[:half], ytr[half:]))
half_te = test_set[0].size(0) // 2
test_set = torch.cat((Xte[:half_te], rotated_Xte[half_te:])), torch.cat((yte[:half_te], yte[half_te:]))
data_handler = ClassificationDataHandler(train_set[0], train_set[1],
                                         test_set[0], test_set[1])

data_dispatcher = CustomDataDispatcher(data_handler, n=5, eval_on_user=False, auto_assign=True)

nodes = PENSNode.generate(
    data_dispatcher=data_dispatcher,
    p2p_net=StaticP2PNetwork(5),
    model_proto=TorchModelHandler(
        net=CIFAR10Net(),
        optimizer= torch.optim.SGD,
        optimizer_params = {
            "lr": 0.01,
            "weight_decay": 0.001
        },
        criterion = F.cross_entropy,
        create_model_mode= CreateModelMode.MERGE_UPDATE,
        batch_size= 8,
        local_epochs= 3),
    round_len=100,
    sync=False,
    n_sampled= 10,
    m_top= 2,
    step1_rounds= 100)

simulator = GossipSimulator(
    nodes = nodes,
    data_dispatcher=data_dispatcher,
    delta=100,
    protocol=AntiEntropyProtocol.PUSH,
    sampling_eval=0.1
)

report = SimulationReport()
simulator.add_receiver(report)
simulator.init_nodes(seed=42)
simulator.start(n_rounds=500)

plot_evaluation([[ev for _, ev in report.get_evaluation(False)]], "Overall test results")