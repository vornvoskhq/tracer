#!/usr/bin/env python3
"""
Paper Processing Pipeline System

A configurable system for running LLM workflows on research papers.
Supports both linear pipelines and feedback loops/iterative workflows.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import PyPDF2
import io

from llm_client_wrapper import LLMClient
from config_utils import load_llm_config, save_llm_config
from chunking_config import ChunkingConfig, ChunkingStrategy, ChunkingTiming, CHUNKING_PRESETS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    PIPELINE = "pipeline"
    FEEDBACK = "feedback"
    ITERATIVE = "iterative"


@dataclass
class AgentConfig:
    """Configuration for a single LLM agent in the workflow."""
    name: str
    prompt_template: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_message: Optional[str] = None
    output_format: str = "text"  # "text", "json", "markdown"
    retry_count: int = 3


@dataclass
class WorkflowConfig:
    """Configuration for the entire processing workflow."""
    name: str
    description: str
    workflow_type: WorkflowType
    agents: List[AgentConfig]
    
    # Pipeline-specific settings
    pass_previous_output: bool = True
    
    # Feedback/Iterative settings
    feedback_agent: Optional[str] = None
    max_iterations: int = 3
    convergence_criteria: Optional[str] = None
    
    # Output settings
    save_intermediate: bool = False
    output_dir: str = "output"
    
    # Chunking configuration
    chunking_config: Optional[ChunkingConfig] = None
    chunking_preset: Optional[str] = None  # Reference to CHUNKING_PRESETS


class PaperProcessor:
    """Main class for processing research papers with configurable LLM workflows."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.llm_client = LLMClient()
        self.workflows: Dict[str, WorkflowConfig] = {}
        self.workflow_summaries: Dict[str, str] = {}  # Store workflow-level summaries
        
        if self.config_path.exists():
            self.load_config()
    
    def load_config(self):
        """Load workflow configurations from JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            for workflow_name, workflow_data in config_data.get("workflows", {}).items():
                # Convert agent configs
                agents = []
                for agent_data in workflow_data.get("agents", []):
                    agents.append(AgentConfig(**agent_data))
                
                # Create workflow config
                workflow_data["agents"] = agents
                workflow_data["workflow_type"] = WorkflowType(workflow_data["workflow_type"])
                
                # Handle chunking configuration
                chunking_config = None
                chunking_preset = workflow_data.pop("chunking_preset", None)
                
                if chunking_preset and chunking_preset in CHUNKING_PRESETS:
                    chunking_config = CHUNKING_PRESETS[chunking_preset]
                    logger.info(f"Using chunking preset '{chunking_preset}' for workflow '{workflow_name}'")
                elif "chunking_config" in workflow_data:
                    # Custom chunking config
                    chunking_data = workflow_data.pop("chunking_config")
                    if isinstance(chunking_data, dict):
                        # Convert string enums back to enum objects
                        if "strategy" in chunking_data:
                            chunking_data["strategy"] = ChunkingStrategy(chunking_data["strategy"])
                        if "timing" in chunking_data:
                            chunking_data["timing"] = ChunkingTiming(chunking_data["timing"])
                        chunking_config = ChunkingConfig(**chunking_data)
                
                workflow_data["chunking_config"] = chunking_config
                workflow_data["chunking_preset"] = chunking_preset
                
                self.workflows[workflow_name] = WorkflowConfig(**workflow_data)
                
            logger.info(f"Loaded {len(self.workflows)} workflow configurations")
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def save_config(self):
        """Save current workflow configurations to JSON file."""
        # Load existing config to preserve LLM settings
        existing_config = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
            except Exception:
                pass
        
        # Update workflows section
        workflow_data = {}
        
        for name, workflow in self.workflows.items():
            workflow_dict = asdict(workflow)
            workflow_dict["workflow_type"] = workflow.workflow_type.value
            workflow_data[name] = workflow_dict
        
        # Update existing config with new workflows
        existing_config["workflows"] = workflow_data
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(existing_config, f, indent=2)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                extracted_text = text.strip()
                logger.info(f"PDF text extraction: {len(extracted_text)} characters from {len(pdf_reader.pages)} pages")
                
                if len(extracted_text) < 100:
                    logger.warning(f"Very short text extracted: '{extracted_text[:200]}...'")
                
                return extracted_text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    async def process_large_input(self, input_text: str, agent: AgentConfig) -> str:
        """Process large input by chunking and summarizing to preserve all information."""
        import gc
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        MAX_CHUNK_SIZE = 30000  # Conservative chunk size
        MAX_CHUNKS_LIMIT = 20  # Safety limit to prevent memory overload
        
        logger.info(f"Agent '{agent.name}': Initial memory usage: {initial_memory:.1f} MB")
        
        if len(input_text) <= MAX_CHUNK_SIZE:
            logger.debug(f"Agent '{agent.name}': Input size {len(input_text)} chars is within limit, no chunking needed")
            return input_text
        
        logger.info(f"Agent '{agent.name}': Large input detected ({len(input_text)} chars), using chunking strategy")
        
        # Calculate chunk parameters more conservatively for memory safety
        overlap = min(2000, MAX_CHUNK_SIZE // 10)  # Smaller overlap for very large chunks
        estimated_chunks = (len(input_text) // (MAX_CHUNK_SIZE - overlap)) + 1
        
        if estimated_chunks > MAX_CHUNKS_LIMIT:
            logger.warning(f"Agent '{agent.name}': Estimated {estimated_chunks} chunks exceeds limit of {MAX_CHUNKS_LIMIT}")
            # Increase chunk size to reduce number of chunks
            adjusted_chunk_size = len(input_text) // MAX_CHUNKS_LIMIT + overlap
            logger.info(f"Agent '{agent.name}': Adjusting chunk size to {adjusted_chunk_size} to limit chunks to {MAX_CHUNKS_LIMIT}")
            MAX_CHUNK_SIZE = adjusted_chunk_size
        
        # Process chunks iteratively to manage memory
        chunk_summaries = []
        start = 0
        chunk_count = 0
        previous_start = -1  # Track previous start position to detect infinite loops
        
        while start < len(input_text) and chunk_count < MAX_CHUNKS_LIMIT:
            # Memory check before processing each chunk
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            logger.debug(f"Agent '{agent.name}': Memory before chunk {chunk_count + 1}: {current_memory:.1f} MB (+{memory_increase:.1f} MB)")
            
            if memory_increase > 500:  # If memory increased by more than 500MB, trigger GC
                logger.warning(f"Agent '{agent.name}': High memory usage detected ({memory_increase:.1f} MB increase), running garbage collection")
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(f"Agent '{agent.name}': Memory after GC: {current_memory:.1f} MB")
            
            # Safety check for infinite loop
            if start == previous_start:
                logger.error(f"Agent '{agent.name}': Infinite loop detected at position {start}, breaking")
                break
            previous_start = start
            
            # Calculate chunk boundaries
            end = min(start + MAX_CHUNK_SIZE, len(input_text))
            
            # Extract chunk (don't store in list to save memory)
            chunk = input_text[start:end]
            chunk_size = len(chunk)
            chunk_count += 1
            
            logger.info(f"Agent '{agent.name}': Processing chunk {chunk_count} ({chunk_size} chars, position {start}-{end})")
            
            # Create a summarization prompt for this chunk
            chunk_prompt = f"""Summarize the key information from this section of a research paper, preserving all important technical details:

{chunk}

Focus on:
- Technical contributions and methods
- Key findings and results  
- Novel approaches or algorithms
- Experimental details
- Conclusions and implications

Provide a comprehensive summary that captures all essential information."""

            try:
                # Process this chunk
                chunk_result = await self._process_single_input(chunk_prompt, agent, f"chunk_{chunk_count}")
                if chunk_result and chunk_result.strip():
                    chunk_summaries.append(f"## Chunk {chunk_count} Summary\n{chunk_result}")
                    logger.debug(f"Agent '{agent.name}': Chunk {chunk_count} processed successfully ({len(chunk_result)} chars)")
                else:
                    logger.warning(f"Agent '{agent.name}': Chunk {chunk_count} produced empty result")
                    
            except Exception as e:
                logger.error(f"Agent '{agent.name}': Error processing chunk {chunk_count}: {e}")
                # Continue with other chunks
                chunk_summaries.append(f"## Chunk {chunk_count} Summary\nError processing chunk: {str(e)}")
            
            # Clear the chunk from memory explicitly
            del chunk
            del chunk_prompt
            
            # Move start position, accounting for overlap
            if end >= len(input_text):
                # We've processed the entire text
                break
            start = end - overlap
            
            # Safety check to prevent infinite loops
            if start < 0:
                start = end
            if start >= len(input_text):
                break
                
            # Safety check for infinite loop
            if chunk_count >= MAX_CHUNKS_LIMIT:
                logger.warning(f"Agent '{agent.name}': Reached chunk limit {MAX_CHUNKS_LIMIT}, stopping")
                break
        
        logger.info(f"Agent '{agent.name}': Processed {chunk_count} chunks, {len(chunk_summaries)} successful summaries")
        
        if not chunk_summaries:
            logger.error(f"Agent '{agent.name}': No successful chunk summaries generated")
            return input_text[:MAX_CHUNK_SIZE]  # Return truncated input as fallback
        
        # Memory check before synthesis
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Agent '{agent.name}': Memory before synthesis: {current_memory:.1f} MB")
        
        # Combine all chunk summaries
        combined_summary = "\n\n".join(chunk_summaries)
        combined_size = len(combined_summary)
        
        logger.info(f"Agent '{agent.name}': Combined summary size: {combined_size} chars")
        
        # Final synthesis if we have multiple chunks
        if len(chunk_summaries) > 1:
            # Check if combined summary is getting too large
            if combined_size > MAX_CHUNK_SIZE * 2:
                logger.warning(f"Agent '{agent.name}': Combined summary ({combined_size} chars) is very large, may cause issues in synthesis")
            
            synthesis_prompt = f"""Synthesize the following chunk summaries into a comprehensive overview of the research paper:

{combined_summary}

Create a unified summary that:
- Integrates information across all sections
- Maintains all technical details
- Preserves the logical flow of the paper
- Highlights key contributions and findings"""
            
            logger.info(f"Agent '{agent.name}': Synthesizing {len(chunk_summaries)} chunks into final summary")
            
            try:
                final_result = await self._process_single_input(synthesis_prompt, agent, "synthesis")
                
                if final_result and final_result.strip():
                    logger.info(f"Agent '{agent.name}': Synthesis successful ({len(final_result)} chars)")
                    return final_result
                else:
                    logger.warning(f"Agent '{agent.name}': Synthesis produced empty result, returning combined summaries")
                    return combined_summary
                    
            except Exception as e:
                logger.error(f"Agent '{agent.name}': Synthesis failed: {e}, returning combined summaries")
                return combined_summary
        else:
            logger.info(f"Agent '{agent.name}': Only one chunk summary, returning directly")
            return chunk_summaries[0] if chunk_summaries else input_text

    async def prepare_workflow_input(self, paper_text: str, workflow: WorkflowConfig) -> str:
        """Prepare input for workflow based on chunking configuration."""
        chunking_config = workflow.chunking_config or CHUNKING_PRESETS.get("light", ChunkingConfig())
        
        if chunking_config.timing != ChunkingTiming.WORKFLOW_START:
            # No workflow-level chunking needed
            return paper_text
            
        logger.info(f"Applying workflow-level chunking with strategy: {chunking_config.strategy}")
        
        if chunking_config.strategy == ChunkingStrategy.NONE:
            return paper_text
            
        elif chunking_config.strategy == ChunkingStrategy.SUMMARIZE_FIRST:
            # Create a workflow-level summarizer agent
            summarizer_agent = AgentConfig(
                name="workflow_summarizer",
                prompt_template=f"""Create a comprehensive summary of this research paper that preserves all important technical details for subsequent analysis:

{{input}}

The summary will be used by other agents for detailed analysis, so include:
- All key technical contributions and methods
- Important experimental results and findings
- Methodology and approach details
- Conclusions and implications
- Any novel concepts or terminology introduced

Target length: {chunking_config.summary_first_max_tokens} tokens.""",
                max_tokens=chunking_config.summary_first_max_tokens,
                temperature=0.1
            )
            
            # Use the existing chunking logic if needed
            summary = await self.process_large_input(paper_text, summarizer_agent)
            logger.info(f"Workflow summary created: {len(summary)} chars from {len(paper_text)} chars")
            
            # Store for potential reference by agents
            workflow_key = f"{workflow.name}"
            self.workflow_summaries[workflow_key] = summary
            
            return summary
            
        elif chunking_config.strategy == ChunkingStrategy.AUTO:
            # Legacy chunking behavior - use existing chunking logic
            temp_agent = AgentConfig(
                name="workflow_chunker",
                prompt_template="{input}",
                max_tokens=1000,
                temperature=0.1
            )
            return await self.process_large_input(paper_text, temp_agent)
            
        else:
            # Fallback for unknown strategies
            logger.warning(f"Strategy {chunking_config.strategy} not implemented, using no chunking")
            return paper_text

    async def _process_single_input(self, prompt_text: str, agent: AgentConfig, context_label: str) -> str:
        """Process a single input with the given agent."""
        # Prepare LLM parameters
        llm_params = {
            "messages": [],
            "temperature": agent.temperature,
            "max_tokens": agent.max_tokens
        }
        
        # Add system message if specified
        if agent.system_message:
            llm_params["messages"].append({
                "role": "system",
                "content": agent.system_message
            })
        
        llm_params["messages"].append({
            "role": "user", 
            "content": prompt_text
        })
        
        # Use agent-specific model or fall back to default
        original_config = None
        if agent.model:
            # Temporarily switch model for this agent
            original_config = load_llm_config()
            new_config = original_config.copy()
            new_config["model"] = agent.model
            save_llm_config(new_config)
        
        try:
            response = await self.llm_client.generate_response(**llm_params)
            return response if response else ""
        except Exception as e:
            logger.warning(f"Agent '{agent.name}' {context_label} failed: {e}")
            return ""
        finally:
            # Restore original config if we changed it
            if agent.model and original_config:
                save_llm_config(original_config)

    async def run_agent(self, agent: AgentConfig, input_text: str, context: Dict[str, Any] = None, workflow_config: WorkflowConfig = None) -> str:
        """Run a single LLM agent with the given input."""
        import gc
        import psutil
        import os
        
        context = context or {}
        
        # Monitor memory at start
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024
        
        logger.info(f"Agent '{agent.name}': Starting with {len(input_text)} chars input, Memory: {start_memory:.1f} MB")
        
        try:
            # Determine if chunking should be applied for this agent
            should_chunk = True
            processed_input = input_text
            
            if workflow_config and workflow_config.chunking_config:
                chunking_config = workflow_config.chunking_config.get_config_for_agent(agent.name)
                should_chunk = chunking_config.should_chunk_for_agent(agent.name, len(input_text))
                
                if not should_chunk:
                    logger.info(f"Agent '{agent.name}': Chunking disabled by configuration")
                    processed_input = input_text
                elif chunking_config.timing == ChunkingTiming.AGENT_LEVEL:
                    logger.info(f"Agent '{agent.name}': Using legacy agent-level chunking")
                    # Use existing chunking logic for legacy behavior
                    processed_input = await self.process_large_input(input_text, agent)
                else:
                    # Workflow-level chunking already applied
                    processed_input = input_text
            else:
                # Default behavior - use existing chunking logic
                processed_input = await self.process_large_input(input_text, agent)
            
            # Check memory after chunking
            post_chunk_memory = process.memory_info().rss / 1024 / 1024
            logger.debug(f"Agent '{agent.name}': After processing, Memory: {post_chunk_memory:.1f} MB (+{post_chunk_memory - start_memory:.1f} MB)")
            
            # Format the prompt template with processed input and context
            prompt = agent.prompt_template.format(
                input=processed_input,
                **context
            )
            
            logger.info(f"Agent '{agent.name}': Final input length: {len(processed_input)}, Prompt length: {len(prompt)}")
            
            # Safety check for prompt size
            if len(prompt) > 100000:  # 100k character limit
                logger.warning(f"Agent '{agent.name}': Very large prompt ({len(prompt)} chars), this may cause issues")
            
        except Exception as e:
            logger.error(f"Agent '{agent.name}': Error in preprocessing: {e}")
            # Fallback to truncated input
            truncated_input = input_text[:30000] if len(input_text) > 30000 else input_text
            prompt = agent.prompt_template.format(
                input=truncated_input,
                **context
            )
            logger.info(f"Agent '{agent.name}': Using fallback truncated input ({len(truncated_input)} chars)")
        
        # Prepare LLM parameters
        llm_params = {
            "messages": [],
            "temperature": agent.temperature,
            "max_tokens": agent.max_tokens
        }
        
        # Add system message if specified
        if agent.system_message:
            llm_params["messages"].append({
                "role": "system",
                "content": agent.system_message
            })
        
        llm_params["messages"].append({
            "role": "user", 
            "content": prompt
        })
        
        # Use agent-specific model or fall back to default
        original_config = None
        if agent.model:
            # Temporarily switch model for this agent
            original_config = load_llm_config()
            new_config = original_config.copy()
            new_config["model"] = agent.model
            save_llm_config(new_config)
        
        logger.info(f"Agent '{agent.name}': Using model: {agent.model or 'default'}")
        
        try:
            for attempt in range(agent.retry_count):
                try:
                    logger.info(f"Agent '{agent.name}': Attempt {attempt + 1}/{agent.retry_count}")
                    response = await self.llm_client.generate_response(**llm_params)
                    
                    if not response or not response.strip():
                        logger.warning(f"Agent '{agent.name}': Received empty response")
                        response = f"No response generated by {agent.name}"
                    
                    logger.info(f"Agent '{agent.name}' completed successfully with {len(response)} character response")
                    return response
                except Exception as e:
                    logger.warning(f"Agent '{agent.name}' attempt {attempt + 1} failed: {e}")
                    if "timeout" in str(e).lower():
                        logger.error(f"Agent '{agent.name}': Timeout error - input may be too large or model too slow")
                    if attempt == agent.retry_count - 1:
                        return f"Agent failed after {agent.retry_count} attempts: {str(e)}"
        finally:
            # Restore original config if we changed it
            if agent.model and original_config:
                save_llm_config(original_config)
        
        return ""
    
    async def run_pipeline_workflow(self, workflow: WorkflowConfig, paper_text: str, paper_path: Path) -> Dict[str, Any]:
        """Run a linear pipeline workflow."""
        results = {
            "workflow": workflow.name,
            "paper": str(paper_path),
            "steps": []
        }
        
        # Apply workflow-level chunking/summarization if configured
        current_input = await self.prepare_workflow_input(paper_text, workflow)
        
        context = {
            "paper_title": paper_path.stem,
            "paper_path": str(paper_path),
            "original_text_length": len(paper_text),
            "processed_input_length": len(current_input)
        }
        
        # Add workflow summary to context if available
        workflow_key = f"{workflow.name}"
        if workflow_key in self.workflow_summaries:
            context["workflow_summary"] = self.workflow_summaries[workflow_key]
        
        for i, agent in enumerate(workflow.agents):
            logger.info(f"Running pipeline step {i+1}/{len(workflow.agents)}: {agent.name}")
            
            # Add previous step outputs to context
            if i > 0 and workflow.pass_previous_output:
                context[f"previous_output"] = current_input
                context[f"step_{i-1}_output"] = results["steps"][-1]["output"]
            
            output = await self.run_agent(agent, current_input, context, workflow)
            
            step_result = {
                "step": i + 1,
                "agent": agent.name,
                "output": output,
                "input_length": len(current_input),
                "output_length": len(output) if output else 0
            }
            results["steps"].append(step_result)
            
            # Update input for next step
            if workflow.pass_previous_output:
                current_input = output
        
        return results
    
    async def run_feedback_workflow(self, workflow: WorkflowConfig, paper_text: str, paper_path: Path) -> Dict[str, Any]:
        """Run a feedback/iterative workflow."""
        results = {
            "workflow": workflow.name,
            "paper": str(paper_path),
            "iterations": []
        }
        
        # Apply workflow-level chunking/summarization if configured
        current_input = await self.prepare_workflow_input(paper_text, workflow)
        
        context = {
            "paper_title": paper_path.stem,
            "paper_path": str(paper_path),
            "original_text_length": len(paper_text),
            "processed_input_length": len(current_input)
        }
        
        # Add workflow summary to context if available
        workflow_key = f"{workflow.name}"
        if workflow_key in self.workflow_summaries:
            context["workflow_summary"] = self.workflow_summaries[workflow_key]
        
        for iteration in range(workflow.max_iterations):
            logger.info(f"Running iteration {iteration+1}/{workflow.max_iterations}")
            
            iteration_result = {
                "iteration": iteration + 1,
                "steps": []
            }
            
            # Run all agents in the iteration
            for i, agent in enumerate(workflow.agents):
                logger.info(f"  Running agent: {agent.name}")
                
                # Add iteration context
                iter_context = context.copy()
                iter_context.update({
                    "iteration": iteration + 1,
                    "current_input": current_input
                })
                
                if iteration > 0:
                    iter_context["previous_iteration"] = results["iterations"][-1]
                
                output = await self.run_agent(agent, current_input, iter_context, workflow)
                
                step_result = {
                    "agent": agent.name,
                    "output": output
                }
                iteration_result["steps"].append(step_result)
            
            results["iterations"].append(iteration_result)
            
            # Update input for next iteration (use output from feedback agent or last agent)
            if workflow.feedback_agent:
                feedback_step = next((s for s in iteration_result["steps"] if s["agent"] == workflow.feedback_agent), None)
                if feedback_step:
                    current_input = feedback_step["output"]
            else:
                current_input = iteration_result["steps"][-1]["output"]
            
            # Check convergence criteria if specified
            if workflow.convergence_criteria and iteration > 0:
                # Simple keyword-based convergence check
                if workflow.convergence_criteria.lower() in current_input.lower():
                    logger.info(f"Convergence criteria met at iteration {iteration + 1}")
                    break
        
        return results
    
    async def process_paper(self, paper_path: Union[str, Path], workflow_name: str) -> Dict[str, Any]:
        """Process a single paper with the specified workflow."""
        paper_path = Path(paper_path)
        
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        workflow = self.workflows[workflow_name]
        
        logger.info(f"Processing paper: {paper_path.name} with workflow: {workflow_name}")
        
        # Extract text from PDF
        paper_text = self.extract_text_from_pdf(paper_path)
        if not paper_text:
            raise ValueError(f"Could not extract text from {paper_path}")
        
        # Run the appropriate workflow type
        if workflow.workflow_type == WorkflowType.PIPELINE:
            results = await self.run_pipeline_workflow(workflow, paper_text, paper_path)
        elif workflow.workflow_type in [WorkflowType.FEEDBACK, WorkflowType.ITERATIVE]:
            results = await self.run_feedback_workflow(workflow, paper_text, paper_path)
        else:
            raise ValueError(f"Unsupported workflow type: {workflow.workflow_type}")
        
        # Save results if requested
        if workflow.save_intermediate:
            output_dir = Path(workflow.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories too
            
            output_file = output_dir / f"{paper_path.stem}_{workflow_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
        
        return results
    
    async def process_papers_batch(self, papers_dir: Union[str, Path], workflow_name: str, pattern: str = "*.pdf") -> List[Dict[str, Any]]:
        """Process multiple papers with the specified workflow."""
        papers_dir = Path(papers_dir)
        paper_files = list(papers_dir.glob(pattern))
        
        logger.info(f"Found {len(paper_files)} papers to process")
        
        results = []
        for paper_file in paper_files:
            try:
                result = await self.process_paper(paper_file, workflow_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {paper_file}: {e}")
                results.append({
                    "paper": str(paper_file),
                    "error": str(e)
                })
        
        return results


async def main():
    """Example usage of the paper processor."""
    processor = PaperProcessor()
    
    # Example: process all papers in the papers directory
    papers_dir = Path("papers/papers")
    if papers_dir.exists():
        results = await processor.process_papers_batch(papers_dir, "summarize_and_analyze")
        print(f"Processed {len(results)} papers")
    else:
        print("Papers directory not found")


if __name__ == "__main__":
    asyncio.run(main())