from hns_projekt2.config import choose_device, seed_everything

def main():
    seed_everything()
    device = choose_device()
    print("DONE!")
