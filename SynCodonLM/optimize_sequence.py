from typing import List, Dict, Optional, Tuple
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

from .utils import synonymous_codons


class CodonOptimizer:
    """Codon optimize sequences with our encoder based mlm.
    
    -Initially, passes in a fully masked sequence of length corresponding to your protein sequence.
    -Each masked position has argmax applied to use the synonymous codon with highest probability for that AA.
    -Because of the use of species token type ids, these vary based off the input embeddings, typically matching max CAI codons.
    -Sampling stops when each position in the optimized sequence matches argmax choice when masked (all codons are optimal given their surroundings).
    
    -The model then performs a left to right search, masking each codon iteratively and replacing with a synonymous codon based on
    either argmax (deterministic) or temp + top-k (optional) + multinomial of softmax (non-deterministic).
    -This will typically not converge to match argmax. Therefore, setting number of rounds relatively low is a good idea. 
    """

    def __init__(self, model_name: str = "jheuschkel/SynCodonLM", device: Optional[str] = None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, config=self.config).to(self.device).eval()

        self.bos_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id
        self.mask_id = self.tokenizer.mask_token_id
        if self.mask_id is None or self.bos_id is None or self.eos_id is None:
            raise ValueError("Tokenizer requires BOS/EOS and MASK tokens.")

        # token_type_ids support
        self.supports_tti = "token_type_ids" in getattr(self.tokenizer, "model_input_names", [])

        # amino acid -> list of codon token ids
        self.aa2ids: Dict[str, List[int]] = {}
        unk = self.tokenizer.unk_token_id
        for aa, codons in synonymous_codons.items():
            ids = []
            for codon in codons:
                cid = self.tokenizer.convert_tokens_to_ids(codon)
                if cid is not None and cid != unk:
                    ids.append(cid)
            self.aa2ids[aa] = ids

    #process input aa by generating a fully masked sequence corresponding to AA length + CLS & SEP

    def _inputs_single_mask(self, seq_ids: List[int], mask_pos: int, species_token_type: Optional[int]) -> Dict[str, torch.Tensor]:
        if not (0 <= mask_pos < len(seq_ids)):
            raise IndexError("mask_pos out of range")
        body = list(seq_ids)
        body[mask_pos] = self.mask_id

        input_ids = torch.tensor([[self.bos_id] + body + [self.eos_id]], dtype=torch.long, device=self.device)
        attn = torch.ones_like(input_ids)
        out = {"input_ids": input_ids, "attention_mask": attn}
        if self.supports_tti and species_token_type is not None:
            out["token_type_ids"] = torch.full_like(input_ids, int(species_token_type))
        return out

    def _inputs_all_mask(self, length: int, species_token_type: Optional[int]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([[self.bos_id] + [self.mask_id] * length + [self.eos_id]], dtype=torch.long, device=self.device)
        attn = torch.ones_like(input_ids)
        out = {"input_ids": input_ids, "attention_mask": attn}
        if self.supports_tti and species_token_type is not None:
            out["token_type_ids"] = torch.full_like(input_ids, int(species_token_type))
        return out

    #select codons with determinism or not..

    @staticmethod
    def _choose(
        logits_row: torch.Tensor,
        candidate_ids: torch.Tensor,
        deterministic: bool,
        temperature: float,
        top_k: Optional[int],
        gen: Optional[torch.Generator],
    ) -> Tuple[int, int]:
        # restrict to synonyms
        cand_logits = logits_row.index_select(0, candidate_ids)
        greedy_idx = int(torch.argmax(cand_logits))
        greedy_id = int(candidate_ids[greedy_idx])

        if deterministic:
            return greedy_id, greedy_id

        scaled = cand_logits / max(1e-6, float(temperature))
        if top_k is not None and 0 < top_k < candidate_ids.numel():
            vals, idx = torch.topk(scaled, k=top_k, dim=0)
            probs = F.softmax(vals, dim=0)
            pick = int(torch.multinomial(probs, 1, generator=gen))
            return int(candidate_ids[idx[pick]]), greedy_id
        else:
            probs = F.softmax(scaled, dim=0)
            pick = int(torch.multinomial(probs, 1, generator=gen))
            return int(candidate_ids[pick]), greedy_id

    # ---------- init ----------

    def _init_greedy(self, aa_seq: str, species_token_type: Optional[int]) -> List[int]:
        """All positions masked; pick per-site argmax among synonyms."""
        L = len(aa_seq)
        with torch.no_grad():
            logits = self.model(**self._inputs_all_mask(L, species_token_type)).logits.squeeze(0)
        ids: List[int] = []
        for i, aa in enumerate(aa_seq):
            cand = self.aa2ids.get(aa)
            if not cand:
                raise ValueError(f"No codons for amino acid '{aa}' at position {i}")
            cand_ids = torch.tensor(cand, dtype=torch.long, device=logits.device)
            # direct argmax over candidate logits (no softmax needed)
            pos = i + 1  # offset for BOS
            pick = int(torch.argmax(logits[pos].index_select(0, cand_ids)))
            ids.append(int(cand_ids[pick]))
        return ids

    #------------- MAIN FUNCTION TO OPTIMIZE =----------

    def optimize(
        self,
        protein_sequence: str,
        max_rounds: int = 100,
        deterministic: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        verbose: bool = False,
        stop_when_argmax_fixed_point: bool = True,
        return_history: bool = False,
        species_token_type: int = 500,
    ) -> Dict:
        """
        Coordinate-wise updates:
          - deterministic=True  -> argmax among synonyms
          - deterministic=False -> multinomial from softmax (seed=9), with temperature/top_k
        token_type_ids (if supported) are filled with species_token_type.
        """
        gen = None
        if not deterministic:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(9)

        curr = self._init_greedy(protein_sequence, species_token_type)
        history = [[self.tokenizer.convert_ids_to_tokens(x) for x in curr]]

        converged = False
        L = len(protein_sequence)

        for r in range(1, max_rounds + 1):
            prev = list(curr)
            greedy_round: List[int] = []

            for i, aa in enumerate(protein_sequence):
                cand = self.aa2ids.get(aa)
                if not cand:
                    greedy_round.append(prev[i])
                    continue

                with torch.no_grad():
                    logits = self.model(**self._inputs_single_mask(prev, i, species_token_type)).logits.squeeze(0)
                row = logits[i + 1]
                cand_ids = torch.tensor(cand, dtype=torch.long, device=row.device)

                chosen, greedy = self._choose(row, cand_ids, deterministic, temperature, top_k, gen)
                prev[i] = chosen
                greedy_round.append(greedy)

            curr = prev

            if stop_when_argmax_fixed_point and curr == greedy_round:
                converged = True
                history.append([self.tokenizer.convert_ids_to_tokens(x) for x in curr])
                if verbose:
                    print(f"[Round {r}] converged")
                break

            step_codons = [self.tokenizer.convert_ids_to_tokens(x) for x in curr]
            history.append(step_codons)
            if verbose:
                print(f"[Round {r}] {' '.join(step_codons)}")

        final_codons = [self.tokenizer.convert_ids_to_tokens(x) for x in curr]
        final_string = "".join(final_codons)  # concatenate without spaces

        result = {
            "final_codons": final_codons,
            "sequence": final_string,
            "converged": converged,
            "rounds": len(history) - 1,
            "reason": "fixed point" if converged else "max rounds",
        }
        if return_history:
            result["history_codons"] = history

        return SimpleNamespace(**result)

