
    assert svc._normalize_dataset_format("robomimic_hdf5") == "hdf5"
    assert svc._normalize_dataset_format("platform_hdf5") == "hdf5"


def test_nut_assembly_job_format_is_hdf5():
    rows = svc.list_datasets_for_api()
    nut_rows = [r for r in rows if r.get("taskType") == "nut_assembly"]
    assert nut_rows, "expected at least one nut_assembly dataset in index"
    for row in nut_rows:
        assert row.get("format") == "hdf5", row.get("sourceJobId")
        assert row.get("datasetFormat") in {None, "robomimic_hdf5", "hdf5"}