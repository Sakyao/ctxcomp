# History Optimizer Module
# This module optimizes history by summarizing past interactions to reduce context length.

import os
import json
import logging
from copy import deepcopy
from typing import Dict, Any, List, Optional, Tuple

try:
    from termcolor import cprint
except ImportError:
    def cprint(text, color=None):
        print(text)

from .base import BaseContextOptimizer


class HistoryOptimizer(BaseContextOptimizer):
    """
    History optimizer that summarizes interaction history to manage context length
    while preserving important information for decision making.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        prompt_dir: str = None, 
        debug_mode: bool = True,
        llm: Optional[Any] = None, # only for local vLLM model
    ):
        """
        Initialize the history optimizer.
        
        Args:
            config: Configuration dictionary containing:
                - prompts: Dictionary of prompt template names (optional)
                - model: Model name for LLM backend
                - temperature: Temperature for LLM generation
                - compressor_type: "full" or "stepwise" compression mode
                - history_summarization_threshold: Token threshold for summarization
            prompt_dir: Directory containing prompt templates
            debug_mode: Whether to enable debug mode
        """
        # Use default prompt directory if not specified
        prompt_dir = config.get("history_prompt_dir", None)
        if prompt_dir is None:
            prompt_dir = 'prompts/context_opt'
            
        super().__init__(config, prompt_dir, debug_mode)
        
        # Set up prompt template names
        self.prompt_config = config.get("prompts", {})
        self.system_template = self.prompt_config.get("prompt_system", "system_prompt")
        self.history_template = self.prompt_config.get("prompt_history_user", "prompt_history")
        
        # Set up summarization threshold
        self.history_summarization_threshold = config.get("history_summarization_threshold", -1)
        # If threshold is -1, it means summarization every time
        
        # Construct system message
        self.system_message = self.construct_system_message(self.system_template)
        self.model_name = config.get("model", "gpt-4o")
        if "lora_name_obs" in config:
            self.lora_name = config["lora_name_obs"]
        else:
            # Fallback to a general Lora name if not specified
            self.lora_name = config.get("lora_name", None)
        self.temperature = config.get("temperature", 0.0)

        if llm:
            self.llm = llm
            # assign the system message to the llm
        else:
            # Lazy import to avoid circular import with agents during module import
            from productive_agents.agents.utils import LLMManager
            self.llm = LLMManager.create_llm(
                self.model_name, '', self.system_message, self.lora_name
            )

        self.use_llmlingua = config.get("use_llmlingua", False)
        
    def process(
        self, 
        task: str, 
        history: List, 
        prev_history_summary: Optional[str] = None,
        raw_history: List = [],
        opt_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[str, int]:
        """
        Summarize history to reduce context length.
        
        Args:
            task: The task given to the agent
            history: The history of agent actions and observations
            prev_history_summary: Previous history summary tuple (summary, last_idx)
            
        Returns:
            Tuple of (summary_text, last_summarized_index)
        """
        # Build the summarization prompt
        prompt, prompt_args = self._build_history_prompt(
            task, history, prev_history_summary
        )
        
        # Generate summary
        # if self.debug_mode:
        #     cprint("History Optimization Prompt", 'red')
        #     cprint(prompt, 'blue')
        if self.use_llmlingua:
            if prev_history_summary:
                _history = prev_history_summary + history
            else:
                _history = history
            raw_response = self.llmlingua_compress_context(_history, ratio=0.2)
        else:
            raw_response = self.llm.generate(prompt, temperature=self.temperature)
            raw_response = raw_response.strip()
        
        # Save interaction to history
        self.add_to_history(self.system_message, prompt, raw_response, prompt_args)
        
        # Parse the response
        keyword = "# History Summary"
        summary = self.parse_output(raw_response, keyword)
        
        if self.debug_mode:
            cprint("History Compression", 'red')
            cprint(raw_response, 'blue')
            
        return summary
    
    def check_summarization_needed(self, history_text: str, prev_history_summary: str=None) -> bool:
        """
        Check if history summarization is needed based on token threshold.
        
        Args:
            history: The history of agent actions and observations
            history_summary: Previous summary tuple (summary, last_idx)
            
        Returns:
            True if summarization is needed, False otherwise
        """
        if self.history_summarization_threshold == -1:
            # Always summarize if threshold is -1
            return True

        if prev_history_summary:
            history_text = f"{prev_history_summary}\n{history_text}"
                    
        # Calculate token count
        n_tokens = self.count_tokens(history_text)
        
        if self.debug_mode:
            print(" ########### History summarization criterion check ###########")
            print(f"        History length: {n_tokens}")
            print(f"        History summarization threshold: {self.history_summarization_threshold}")
            
        return n_tokens > self.history_summarization_threshold
    
    def _build_history_prompt(self, task: str, history: List, prev_history_summary: Optional[Tuple]) -> str:
        """
        Build the prompt for history summarization.
        
        Args:
            task: The task description
            history: List of (action, observation) pairs
            current_app: Current app being used
            available_apps: Available apps dictionary
            prev_history_summary: Previous summary tuple (summary, last_idx)
            
        Returns:
            Formatted prompt string
        """
        prompt_args = {"task": task}
        
        # Handle previous history summary
        if prev_history_summary:
            prev_history_text = f"<PREVIOUS_SUMMARY>\n{prev_history_summary}\n</PREVIOUS_SUMMARY>"
            prompt_args["prev_summary"] = prev_history_text
        else:
            prompt_args["prev_summary"] = ''
        
        # Full mode: include all history
        prompt_args["history"] = history
        template_name = self.history_template
        
        # Render the prompt
        prompt = self.render_template(template_name, **prompt_args)
        
        if self.debug_mode:
            logging.debug("History Optimizer Prompt:\n" + prompt)
            
        return prompt, prompt_args

    def dump_history(self, output_dir: str):
        """
        Dump the observation optimizer's history to a JSON file.
        
        Args:
            output_dir: Directory to save the history file
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        history_file = os.path.join(output_dir, "history_optimizer_history.json")
        
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"History optimizer history dumped to {history_file}")


class HistoryOptimizerV2(HistoryOptimizer):
    """
    History optimizer that summarizes interaction history to manage context length
    while preserving important information for decision making.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        prompt_dir: str = None, 
        debug_mode: bool = True,
        llm: Optional[Any] = None, # only for local vLLM model
    ):
        """
        Initialize the history optimizer v2.
        Reuse the KV-cache of the agent LLM as much as possible. 
        
        Args:
            config: Configuration dictionary containing:
                - prompts: Dictionary of prompt template names (optional)
                - model: Model name for LLM backend
                - temperature: Temperature for LLM generation
                - compressor_type: "full" or "stepwise" compression mode
                - history_summarization_threshold: Token threshold for summarization
            prompt_dir: Directory containing prompt templates
            debug_mode: Whether to enable debug mode
        """
        super().__init__(config, prompt_dir, debug_mode, llm)
        # The rest is handled in the parent class.
        self.system_message = ''

    def process(
        self, 
        task: str, 
        history: List, 
        prev_history_summary: Optional[str] = None,
        raw_history: List = [],
        opt_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[str, int]:
        """
        Summarize history to reduce context length.
        
        Args:
            task: The task given to the agent
            history: The history of agent actions and observations
            prev_history_summary: Previous history summary tuple (summary, last_idx)
            
        Returns:
            Tuple of (summary_text, last_summarized_index)
        """
        # Build the summarization prompt
        prompt, prompt_args = self._build_history_prompt(
            task, raw_history
        )
        
        # Generate summary
        # if self.debug_mode:
        #     cprint("History Optimization Prompt", 'red')
        #     cprint(prompt, 'blue')
        raw_response = self.llm.generate(prompt, temperature=self.temperature)
        raw_response = raw_response.strip()

        # Just for logging purpose
        if prev_history_summary:
            prompt_args["prev_summary"] = prev_history_summary
        else:
            prompt_args["prev_summary"] = ''
        
        # Just for logging purpose
        prompt_args["history"] = history
        
        # Save interaction to history
        self.add_to_history(self.system_message, prompt, raw_response, prompt_args)
        
        # Parse the response
        keyword = "# History Summary"
        summary = self.parse_output(raw_response, keyword)
        
        if self.debug_mode:
            cprint("History Compression", 'red')
            cprint(raw_response, 'blue')
            
        return summary
    
    
    def _build_history_prompt(self, task: str, raw_history: List) -> str:
        """
        Build the prompt for history summarization.
        
        Args:
            task: The task description
            raw_history: List of (action, observation) pairs
            
        Returns:
            Formatted prompt string
        """
        prompt_args = {"task": task}
        template_name = self.history_template
        
        # Render the prompt
        prompt = self.render_template(template_name, **prompt_args)

        whole_prompt = deepcopy(raw_history)

        whole_prompt[-1]["content"] += "\n[COMPRESSION START]\n" + prompt
        if self.debug_mode:
            logging.debug("History Optimizer Prompt:\n" + prompt)
            
        return whole_prompt, prompt_args

try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    cosine_similarity = None
    np = None
    print("Please install scikit-learn and numpy for HistoryRetriever functionality.")

class HistoryRetriever(HistoryOptimizer):
    """
    History optimizer that summarizes interaction history to manage context length
    while preserving important information for decision making.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        prompt_dir: str = None, 
        debug_mode: bool = True,
        llm: Optional[Any] = None, # only for local vLLM model
    ):
        """
        Initialize the history optimizer v2.
        Reuse the KV-cache of the agent LLM as much as possible. 
        
        Args:
            config: Configuration dictionary containing:
                - prompts: Dictionary of prompt template names (optional)
                - model: Model name for LLM backend
                - temperature: Temperature for LLM generation
                - compressor_type: "full" or "stepwise" compression mode
                - history_summarization_threshold: Token threshold for summarization
            prompt_dir: Directory containing prompt templates
            debug_mode: Whether to enable debug mode
        """
        super().__init__(config, prompt_dir, debug_mode, llm)
        # The rest is handled in the parent class.
        self.system_message = ''
        # Define embedding model for retrieval here
        from productive_agents.subtrate_api import AzureOpenAIEmbeddings
        self.embedding_model = AzureOpenAIEmbeddings(model_name="text-embedding-3-large")
        # text -> vector, keyed by content rather than position. A positional list
        # is only valid while the candidate pool grows by appending; the caller
        # de-duplicates that pool, which can shorten it, so the cache must be
        # addressable by content. See retrieve().
        self.embedding_cache = {}
        # Cosine similarity above which two candidate turns are treated as the
        # same interaction for retrieval. High enough that genuinely different
        # steps stay distinct, low enough to collapse near-identical repeats that
        # differ only in a token or id. See retrieve().
        self.duplicate_similarity = config.get("retrieval_duplicate_similarity", 0.95)

    def convert_to_text(self, history: List) -> str:
        """
        Convert history list to a single text string.
        
        Args:
            history: The history of agent actions and observations
            
        Returns:
            Concatenated history text
        """
        history_text_list = []
        history_text = ""
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_text += f"{role.capitalize()}: {content}\n"
            if role == "user":
                history_text_list.append(history_text)
                history_text = ""
        return history_text_list

    def retrieve(self, history: List, last_turn: List, n_turns: int) -> List:
        """
        Retrieve the n_turns most relevant turns from the history.

        Turns are ranked by cosine similarity to `last_turn`, but the selection
        then skips turns whose observation has already been taken and turns whose
        observation is the one the agent is currently staring at. Selecting purely
        by similarity makes the baseline self-destructive: an agent that repeats
        an action gets that same action back as its own top hit, the copy goes
        into the rebuilt context, and the repetition grows on every rebuild until
        the window is nothing but copies of one action. Measured on AppWorld
        before this change: 80% of the final context was a single repeated
        observation (FIFO 0%, no compression 3%), 166/168 tasks ran into the
        max_iter cap, and the row scored 1.2%.

        Args:
            history: The history of agent actions and observations
            last_turn: The current turn, used as the retrieval query
            n_turns: Number of turns to retrieve

        Returns:
            The selected turns in chronological order, so the caller can append
            them to the new session as an ordinary transcript.
        """
        query = self.convert_to_text(last_turn)[0]
        history_text_list = self.convert_to_text(history)

        # Cache embeddings by content rather than by position. A positional cache
        # is only correct while `history` keeps growing by appending, because it
        # diffs the new text list against everything accumulated so far and the
        # similarity vector is then taken over the whole cache. Keying on content
        # keeps the ranking tied to the list actually passed in, whatever its
        # length.
        missing = [text for text in history_text_list if text not in self.embedding_cache]
        if missing:
            for text, vector in zip(missing, self.embedding_model.embed_documents(missing)):
                self.embedding_cache[text] = vector

        query_embedding = self.embedding_model.embed_query(query)

        # similarity search
        similarities = cosine_similarity(
            [query_embedding],
            np.array([self.embedding_cache[text] for text in history_text_list])
        )[0]

        # Walk the ranking from most to least similar and keep the first n_turns
        # turns that are not repeats of something already taken.
        #
        # Two kinds of repeat have to be rejected, and text comparison only
        # catches the first. When the agent is stuck it re-issues the same call
        # and each call returns a freshly minted value -- a new access token, a
        # new record id -- so consecutive observations are never byte-identical
        # even though they carry no new information. With text comparison alone
        # the median rebuilt context stayed 80% one repeated observation and 82%
        # of tasks hit the max_iter cap. Comparing the cached embeddings catches
        # those near-repeats for free, because every candidate is already
        # embedded for the ranking.
        #
        # Seeding with the current turn also stops the agent from being handed
        # back the step it is in the middle of repeating. If every candidate is a
        # repeat, the selection comes back empty and the caller rebuilds the
        # session from the task prompt plus the preserved turns alone -- a clean
        # slate, which is what breaks the loop.
        current_observation = last_turn[-1]['content'] if last_turn else None
        seen_observations = {current_observation}

        # Seed with the current turn's own vector. It must come from
        # `query_embedding` rather than the cache: the cache is built from
        # `history`, and the caller's pool deliberately excludes the current turn,
        # so a cache lookup here raises KeyError. That exception is swallowed by
        # the caller's handler and surfaces only as "History optimization failed",
        # which silently disables compression for the rest of the task.
        taken_vectors = [np.array(query_embedding)]

        selected = []
        for i in np.argsort(similarities)[::-1]:
            if len(selected) >= n_turns:
                break
            observation = history[i*2+1]['content']
            if observation in seen_observations:
                continue
            vector = np.array(self.embedding_cache[history_text_list[int(i)]])
            if taken_vectors:
                closest = float(cosine_similarity([vector], taken_vectors)[0].max())
                if closest >= self.duplicate_similarity:
                    continue
            seen_observations.add(observation)
            taken_vectors.append(vector)
            selected.append(int(i))

        # Chronological order, so the rebuilt session reads as a transcript.
        selected = sorted(selected)
        print(f"Retrieved turns indices: {selected}")
        retrieved_turns = []
        for i in selected:
            retrieved_turns.append(history[i*2])
            retrieved_turns.append(history[i*2+1])
        return retrieved_turns

    def dump_history(self, output_dir: str):
        return