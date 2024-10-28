task_map = {
    "promoter": {
        "title":  "Core promoter prediction",
        "desc":   "<p>Predict whether the input sequence is a active core promoter.</p>",
        "labels": ["Not promoter", "Core promoter"],
        "datatype": "single_classification"
    },
    "H3K27ac": {
        "title":  "Histone modification (H3K27ac) prediction",
        "desc":   "<p>Predict whether the input sequence is from H3K27ac histone modification regions.</p>",
        "labels": ["Not H3K27ac", "H3K27ac"],
        "datatype": "single_classification"
    },
    "H3K27me3": {
        "title": "Histone modification (H3K27me3) prediction",
        "desc": "<p>Predict whether the input sequence is from H3K27me3 histone modification regions.</p>",
        "labels": ["Not H3K27me3", "H3K27me3"],
        "datatype": "single_classification"
    },
    "H3K4me3": {
        "title": "Histone modification (H3K4me3) prediction",
        "desc": "<p>Predict whether the input sequence is from H3K4me3 histone modification regions.</p>",
        "labels": ["Not H3K4me3", "H3K4me3"],
        "datatype": "single_classification"
    },
    "conservation": {
        "title":  "Sequence conservation prediction",
        "desc":   "<p>Predict whether the input sequence is conserved sequence.</p>",
        "labels": ["Not conserved", "Conserved"],
        "datatype": "single_classification"
    },
    "lncRNAs": {
        "title": "Putative IncRNAs prediction",
        "desc": "<p>Predict whether the input sequence is an IncRNA.</p>",
        "labels": ["Not lncRNA", "lncRNA"],
        "datatype": "single_classification"
    },
    "open_chromatin": {
        "title": "Open chromation regions prediction",
        "desc": "<p>Predict whether the input sequence is from open chromatin regions (detected by DNase-seq, ATAC-seq, MH-seq, etc.).</p>",
        "labels": ["Not open chromatin", "Full open chromatin", "Partial open chromatin"],
        "datatype": "multi_classification"
    },
    "promoter_strength_leaf": {
        "title": "Promoters strength prediction",
        "desc": "<p>Predict the (core) promoter strength in tobacco leaves based on STARR-seq data.</p>",
        "labels": ["Promoter strength in tobacco leaves"],
        "datatype": "regression"
    },
    "promoter_strength_protoplast": {
        "title": "Promoters strength prediction",
        "desc": "<p>Predict the (core) promoter strength in maize protoplasts based on STARR-seq data.</p>",
        "labels": ["Promoter strength in maize protoplasts"],
        "datatype": "regression"
    }
}
