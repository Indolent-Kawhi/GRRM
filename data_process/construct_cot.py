import json
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from prompt import SYN_SYSTEM_PROMPT
from tqdm import tqdm
import threading
import time
import random
import argparse

random.seed(42)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, root: str, dataset: str, api_key: str, model_name: str = "qwen3", 
                 base_url: str = "",
                 ports: Optional[List[int]] = None,
                 min_history_length: int = 4,
                 tail_discard: int = 2,
                 max_history_window: int = 20,
                 num_samples: Optional[int] = None,
                 history_length_range: Optional[Tuple[int, int]] = None):
        """
        api_key: API key
        model_name: Model name, used for output file naming
        min_history_length: Minimum history length required to generate a sample (history >= this value to predict next target)
        tail_discard: Number of interactions to discard from the end of the sequence
        max_history_window: Maximum history window length (truncate to recent part if exceeded)
        num_samples: Limit the number of samples to process, None means process all
        history_length_range: History length filter range (min_len, max_len), None means no filter
        """
        self.dataset = dataset
        self.api_key = api_key if api_key else ""
        self.model_name = model_name
        self.base_url = base_url
        self.ports = ports or []
        self.port_index = 0
        self.port_lock = threading.Lock()
        self.min_history_length = min_history_length
        self.tail_discard = tail_discard
        self.max_history_window = max_history_window
        self.num_samples = num_samples
        self.history_length_range = history_length_range
        
        # Initialize client(s) based on ports
        if self.ports:
            self.clients = [OpenAI(api_key="fake-api-key", base_url=f"http://localhost:{port}/v1") for port in self.ports]
            logger.info(f"Initialized port pool mode, ports: {self.ports}")
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info(f"Initialized single endpoint mode, URL: {base_url}")

        self.data_path = Path(f'{root}/{dataset}')
        self.output_file = Path(f'{root}/{dataset}', f'result_{model_name}.jsonl')
        
        # Load data
        self.inters = self._load_json_file(self.data_path / f'{dataset}.inter.json')
        self.items = self._load_json_file(self.data_path / f'{dataset}.item.json')
        self.reviews = self._load_json_file(self.data_path / f'{dataset}.review.json')
        
        # Load processed samples directly from result file
        self.processed_samples = self._load_processed_samples_from_results()

    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            raise

    def _load_processed_samples_from_results(self) -> set:
        """Load processed samples directly from result file"""
        processed_samples = set()
        if not self.output_file.exists():
            logger.info("Result file does not exist, start from scratch")
            return processed_samples
        
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        user_id = str(obj.get("user_id", ""))
                        start_index = obj.get("history_start_index")
                        end_index = obj.get("history_end_index")
                        history_length = obj.get("history_length")
                        
                        if all(x is not None for x in [user_id, start_index, end_index, history_length]):
                            sample = (user_id, start_index, end_index, history_length)
                            processed_samples.add(sample)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Result file line {line_no} format error, skipped: {e}")
                        continue
            
            logger.info(f"Loaded {len(processed_samples)} processed samples from result file")
            
        except Exception as e:
            logger.error(f"Failed to load processed samples from result file: {e}")
        
        return processed_samples

    def _get_next_client(self) -> OpenAI:
        """Get the next available client (round-robin)"""
        if not self.ports:
            return self.client
        
        with self.port_lock:
            client = self.clients[self.port_index]
            self.port_index = (self.port_index + 1) % len(self.clients)
            return client
    
    def collect_all_samples(self) -> List[Tuple[str, int, int, int]]:
        """
        Collect all possible samples: returns a list of (user_id, start_index, end_index, history_length)
        """
        all_samples = []
        for user_id, seq in self.inters.items():
            usable_len = len(seq) - self.tail_discard
            if usable_len >= self.min_history_length + 1:
                # target_idx range: [min_history_length, usable_len-1]
                for target_idx in range(self.min_history_length, usable_len):
                    history_seq_full = seq[:-self.tail_discard][:target_idx] if self.tail_discard > 0 else seq[:target_idx]
                    
                    # Apply window truncation
                    if len(history_seq_full) > self.max_history_window:
                        history_seq = history_seq_full[-self.max_history_window:]
                        history_start_index = target_idx - self.max_history_window
                    else:
                        history_seq = history_seq_full
                        history_start_index = 0
                    
                    history_end_index = target_idx - 1
                    history_length = len(history_seq)
                    
                    # Apply history length filter
                    if self.history_length_range:
                        min_len, max_len = self.history_length_range
                        if not (min_len <= history_length <= max_len):
                            continue
                    
                    all_samples.append((user_id, history_start_index, history_end_index, history_length))
        
        logger.info(f"Total samples collected: {len(all_samples)}")
        if self.history_length_range:
            logger.info(f"History length filter range: {self.history_length_range}")
        return all_samples

    def select_samples_to_process(self, all_samples: List[Tuple[str, int, int, int]]) -> List[Tuple[str, int, int, int]]:
        """
        Select samples to process from all samples, excluding already processed samples
        """
        # Filter processed samples
        unprocessed_samples = [sample for sample in all_samples if sample not in self.processed_samples]
        logger.info(f"Total samples: {len(all_samples)}, processed: {len(self.processed_samples)}, to process: {len(unprocessed_samples)}")
        
        if not unprocessed_samples:
            logger.info("All samples have been processed")
            return []
        
        if self.num_samples is None or self.num_samples >= len(unprocessed_samples):
            selected_samples = unprocessed_samples[:]
        else:
            # Randomly select specified number of samples
            selected_samples = random.sample(unprocessed_samples, self.num_samples)
            logger.info(f"Randomly selected {len(selected_samples)} samples from {len(unprocessed_samples)} unprocessed samples")
        
        return selected_samples

    def process_single_sample(self, user_id: str, start_index: int, end_index: int, history_length: int, max_retries: int = 3) -> Dict[str, Any]:
        """Process a single sample"""
        for attempt in range(max_retries):
            try:
                full_seq = self.inters.get(user_id, [])
                trimmed_seq = full_seq[:-self.tail_discard] if self.tail_discard > 0 else full_seq[:]
                
                target_idx = end_index + 1
                target_item = trimmed_seq[target_idx]
                
                # Extract history sequence based on start_index and end_index
                history_seq = trimmed_seq[start_index:end_index+1]
                
                # Build history text
                history_lines = []
                for i, item_id in enumerate(history_seq):
                    item_info = self.items.get(str(item_id), {})
                    block = [f"Item {i + 1}:"]
                    block.extend([f"{k}: {v if (v != '' and v != '.') else 'N/A'}" for k, v in item_info.items() if k in ['title', 'description']])
                    history_lines.append("\n".join(block))
                history_text = "\n\n".join(history_lines)
                
                # Build target text
                tgt_info = self.items.get(str(target_item), {})
                target_text = "\n".join([f"{k}: {v}" for k, v in tgt_info.items() if k in ['title', 'description']])
                
                user_prompt = (
                    f"User purchase history:\n\n{history_text}\n\n"
                    f"Next Item:\n{target_text}\n\n"
                )

                messages = [
                    {"role": "system", "content": SYN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
                
                client = self._get_next_client()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    reasoning_effort='low'
                )
                
                return {
                    "user_id": user_id,
                    "target_index": target_idx,
                    "history_length": history_length,
                    "history_start_index": start_index,
                    "history_end_index": end_index,
                    "user_prompt": user_prompt,
                    "assistant_response": response.choices[0].message.content
                }
            except Exception as e:
                logger.warning(f"Sample ({user_id}, {start_index}, {end_index}) attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.error(f"Sample ({user_id}, {start_index}, {end_index}) final failure: {e}")
                    raise

    def save_results(self, results: List[Dict[str, Any]]):
        """Save multiple results"""
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def process_batch_samples(self, samples: List[Tuple[str, int, int, int]], workers: int = 5):
        """Batch process samples"""
        logger.info(f"Start processing {len(samples)} samples")
        
        processed_samples = 0
        progress_bar = tqdm(total=len(samples), desc="Processing samples")
        
        # Simplified signal handling
        import signal
        def signal_handler(signum, frame):
            logger.info("Received exit signal, exiting program")
            import sys
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        def process_with_save(sample: Tuple[str, int, int, int]):
            user_id, start_index, end_index, history_length = sample
            try:
                result = self.process_single_sample(user_id, start_index, end_index, history_length)
                if result:
                    self.save_results([result])
                    # Only need to update the in-memory set, no checkpoint saving needed
                    self.processed_samples.add(sample)
                
                return True
            except Exception as e:
                logger.error(f"Sample ({user_id}, {start_index}, {end_index}) failed: {e}")
                return False
        
        # Concurrent processing
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_sample = {executor.submit(process_with_save, sample): sample for sample in samples}
                
                successful_samples = 0
                failed_samples = 0

                for future in as_completed(future_to_sample):
                    success = future.result()
                    if success:
                        successful_samples += 1
                    else:
                        failed_samples += 1
                    
                    processed_samples += 1
                    progress_bar.update(1)
        finally:
            progress_bar.close()
            logger.info(f"Batch processing finished: successful samples {successful_samples}, failed samples {failed_samples}")

    def run(self, workers: int = 5):
        """Run main processing flow"""
        logger.info("Start processing data...")
        
        # Collect all samples
        all_samples = self.collect_all_samples()
        if not all_samples:
            logger.info("No samples to process")
            return
        
        # Select samples to process
        selected_samples = self.select_samples_to_process(all_samples)
        
        logger.info(f"Start processing {len(selected_samples)} samples")
        self.process_batch_samples(selected_samples, workers)
        
        logger.info("Data processing finished!")

def main():
    parser = argparse.ArgumentParser(description='Data processing program')
    
    # Required arguments
    parser.add_argument('--root', type=str, required=True, 
                        help='Dataset root directory')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset name')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key')

    # Optional arguments
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name, used for output file naming (default: qwen3)')
    parser.add_argument('--base_url', type=str, default=None,
                        help='API base URL')
    parser.add_argument('--ports', type=int, nargs='*', 
                        default=None)
    parser.add_argument('--workers', type=int, default=128,
                        help='Max number of workers')
    parser.add_argument('--min_history_length', type=int, default=4,
                        help='Minimum history length')
    parser.add_argument('--tail_discard', type=int, default=2,
                        help='Number of tail interactions to discard (default: 2)')
    parser.add_argument('--max_history_window', type=int, default=20,
                        help='Max history window length (default: 20)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Limit the number of samples to process, process all if not specified')
    parser.add_argument('--history_length_min', type=int, default=4,
                        help='History length filter min value (default: 4)')
    parser.add_argument('--history_length_max', type=int, default=20,
                        help='History length filter max value (default: 20)')
    parser.add_argument('--no_history_filter', action='store_true',
                        help='Disable history length filter')
    
    args = parser.parse_args()
    
    # Handle ports
    ports = args.ports if args.ports else None
    
    # Handle history length filter
    history_length_range = None if args.no_history_filter else (args.history_length_min, args.history_length_max)
    
    # Create processor
    processor = DataProcessor(
        root=args.root,
        dataset=args.dataset,
        api_key=args.api_key,
        model_name=args.model_name,
        base_url=args.base_url,
        ports=ports,
        min_history_length=args.min_history_length,
        tail_discard=args.tail_discard,
        max_history_window=args.max_history_window,
        num_samples=args.num_samples,
        history_length_range=history_length_range
    )
    
    processor.run(workers=args.workers)

if __name__ == "__main__":
    main()