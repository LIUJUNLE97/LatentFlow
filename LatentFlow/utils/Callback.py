
class TrainingLogger:
    def __init__(self, print_every=10):
        self.print_every = print_every

    def on_epoch_end(self, epoch, train_loss, current_loss_train, val_loss, current_loss_val):
        if epoch % self.print_every == 0:
            if val_loss is not None:
                print(f"[Epoch {epoch:04d}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}, current epoch train loss: {current_loss_train:.6f}, current epoch val loss: {current_loss_val:.6f}")
            else:
                print(f"[Epoch {epoch:04d}] Train Loss: {train_loss:.6f}")
def make_dir():
    import os
    dir = os.getcwd()

    folder = 'LatentFlow/results'
    base_dir = f'{dir}/{folder}/CVAE'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        # log_path = f'{base_dir}/log.txt'
    else:
        i = 1
        while os.path.exists(f'{base_dir}_{i}'):
            i += 1
        base_dir = f'{base_dir}_{i}'
        os.makedirs(base_dir)
        # log_path = f'{base_dir}/log.txt'
    return base_dir

def get_latest_dir():
    import os
    dir = os.getcwd()
    folder = 'LatentFlow/results'
    base_root = f'{dir}/{folder}'

    dirs = [d for d in os.listdir(base_root) if d.startswith('CVAE_') and os.path.isdir(os.path.join(base_root, d))]
    if not dirs:
        raise FileNotFoundError(f"No CVAE directories found in {base_root}.")

    latest_dir = max(dirs, key=lambda d: int(d.split('_')[-1]))
    return os.path.join(base_root, latest_dir)

    