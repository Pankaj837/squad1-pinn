from squad1.contracts.candidate import CandidateDesign, ProjectionResult, check_design_tensor
from squad1.contracts.channels import (
    ChannelNormalizer,
    ChannelSpec,
    DomainSpec,
    available_domains,
    get_domain,
    register_domain,
)
from squad1.contracts.cond import CondSpec

__all__ = [
    "CandidateDesign",
    "ChannelNormalizer",
    "ChannelSpec",
    "CondSpec",
    "DomainSpec",
    "ProjectionResult",
    "available_domains",
    "check_design_tensor",
    "get_domain",
    "register_domain",
]
