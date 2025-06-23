# main.py
import hydra
from omegaconf import DictConfig, OmegaConf
from src.util.classes import HDFStoreManager
from src.models.training import train_CLM_base

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg : DictConfig) -> None:
    
    loss = train_CLM_base(cfg)
    
    HDFStoreManager.close_all()
    return loss

if __name__ == "__main__":
    main()