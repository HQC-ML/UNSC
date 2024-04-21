import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy


class LossContour():
    def __init__(self,) -> None:
        pass
    
    @staticmethod
    def flatten_parameters(model):
        return torch.cat([param.flatten() for param in model.parameters()])

    @staticmethod
    def assign_params(model, w):
        offset = 0
        for name, parameter in model.named_parameters():
            param_size = parameter.nelement()
            if 'bn' not in name:
                parameter.data = w[offset : offset + param_size].reshape(parameter.shape)
            offset += param_size

    @staticmethod
    def flatten_gradients(model):
        return np.concatenate(
            [
                param.grad.detach().cpu().numpy().flatten()
                if param.grad is not None
                else np.zeros(param.nelement())
                for param in model.parameters()
            ]
        )

    @staticmethod
    def get_xy(point, origin, vector_x, vector_y):
        """Return transformed coordinates of a point given parameters defining coordinate
        system.
        Args:
            point: point for which we are calculating coordinates.
            origin: origin of new coordinate system
            vector_x: x axis of new coordinate system
            vector_y: y axis of new coordinate system
        """
        return np.array(
            [
                torch.dot(point - origin, vector_x).item(),
                torch.dot(point - origin, vector_y).item(),
            ]
        )
    
    @staticmethod
    def eval_fn(model, test_loader):
        model.eval()
        correct = 0
        loss = 0
        total = 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(test_loader):
                x = x.cuda()
                y = y.cuda()
                pred_y = model(x)
                predicted = torch.argmax(pred_y.data, 1)
                batch_loss = F.cross_entropy(pred_y, y)
                loss += batch_loss.item()
                total += y.size(0)
                correct += (predicted == y).sum().item()
            acc = correct / total
            loss = loss / total
        return {"loss": loss, "accuracy": acc}


    def calculate_loss_contors(self, model1, model2, dataloader, proj_mat, device='cuda', granularity=20, margin=0.2):
        in_plane_diff = {}
        out_plane_diff = {}
        diff = {}
        model1 = model1.cpu()
        model2 = model2.cpu()
        for (n1,p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            diff[n1] = p2-p1
            
        # Define x and y axis
        kk=0
        for k, (m, params) in enumerate(diff.items()):
            if len(params.shape) > 1:
                sz = params.shape[0]
                in_plane_diff[m] = torch.mm(params.view(sz,-1), proj_mat[kk]).view(params.size())
                out_plane_diff[m] = params - in_plane_diff[m]
                kk +=1

        x_list = {}
        x_coeffs = {}
        for k, v in in_plane_diff.items():
            x_list[k] = v/torch.norm(v)
            x_coeffs[k] = torch.norm(v)
            
        y_list = {}
        y_coeffs = {}
        for k, v in out_plane_diff.items():
            y_list[k] = v/torch.norm(v)
            y_coeffs[k] = torch.norm(v)

        alphas = np.linspace(0.0 - margin, 2.0 + margin, granularity)
        betas = np.linspace(0.0 - margin, 2.0 + margin, granularity)
        losses = np.zeros((granularity, granularity))
        accuracies = np.zeros((granularity, granularity))
        grid = np.zeros((granularity, granularity, 2))

        progress = tqdm(total=granularity * granularity)
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                intermediate_model = deepcopy(model1)
                for m, params in intermediate_model.named_parameters():
                    if 'feature' in m and len(params.size())!=1:
                        params.data = params.data + alpha * x_list[m].data + beta * y_list[m].data
                        # params.data = params.data + alpha * in_plane_diff[m].data + beta * in_plane_diff[m].data
    
                metrics = self.eval_fn(intermediate_model.cuda(), dataloader)
                grid[i, j] = [alpha, beta]
                losses[i, j] = metrics["loss"]
                accuracies[i, j] = metrics["accuracy"]
                progress.update()
                del intermediate_model
        progress.close()
        return {
            "grid": grid.tolist(),
            "losses": losses.tolist(),
            "accuracies": accuracies.tolist(),
        }