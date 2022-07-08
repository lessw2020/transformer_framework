import torch

gigabyte_size = 1073741824
megabyte_size = 1048576


def format_to_gb(item, precision=4):
    """quick function to format numbers to gigabyte and round to (default) 4 digit precision"""
    metric_num = item / gigabyte_size
    metric_num = round(metric_num, ndigits=precision)
    return metric_num


class Memory_Maximizer:
    def __init__(
        self,
    ):
        current_free, total_gpu_mem = torch.cuda.mem_get_info()
        m_total_gpu_mem = format_to_gb(total_gpu_mem)

        m_reserved_memory_list = []

        m_total_ooms = 0
        m_num_retries = 0
        m_max_reserved = 0

    def start(self):
        """start memory tracking, reset any current info"""

        torch.cuda.reset_peak_memory_stats()
        m_reserved_memory_list = []
        m_num_retries = 0
        m_total_ooms = 0
        m_max_reserved = 0

        print(f"reserved and peak memory stats reset, ready to track")

    def update(
        self,
    ):
        """update reserved memory for this epoch"""
        updated_reserved = torch.cuda.memory_reserved()
        updated_reserved = format_to_gb(updated_reserved)
        self.m_reserved_memory_list.append(updated_reserved)

    def stop(
        self,
    ):
        """end of training...get various stats and display"""

        cuda_max_reserved = format_to_gb(torch.cuda.max_memory_reserved())
        print(f"--> cuda max reserved memory = {cuda_max_reserved}")
        print(
            f"--> max reserved percentage = {round(cuda_max_reserved/self.m_available_gpu_mem,4)}"
        )

        cuda_info = torch.cuda.memory_stats()

        self.m_num_retries = cuda_info.get("num_alloc_retries", 0)
        self.m_cuda_ooms = cuda_info.get("num_ooms", 0)

        print(f"cuda retries = {self.m_num_retries}")
        print(f"cuda OOM = {self.m_cuda_ooms}")
        if self.m_num_retries > 0:
            print(
                f"--> Recommend decreasing batch size...cuda retries can greatly degrade perf!"
            )

    def summary(
        self,
    ):
        pass
