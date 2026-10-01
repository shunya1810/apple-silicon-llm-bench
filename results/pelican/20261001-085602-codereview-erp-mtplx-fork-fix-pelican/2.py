if hidden_variant is not None:
    query.append("AND hidden_variant = ?")
    params.append(str(hidden_variant))
if template_hash is not None:
    query.append("AND template_hash = ?")
    params.append(str(template_hash))
if draft_head_identity is not None:
    query.append("AND draft_head_identity = ?")
    params.append(str(draft_head_identity))
if policy_fingerprint is not None:
    query.append("AND policy_fingerprint = ?")
    params.append(str(policy_fingerprint))
