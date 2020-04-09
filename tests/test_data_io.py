from basictools import data_io as dio
import os
import numpy as np


emd1 = "examples/data/SuperX-HAADF 1913.emd"
d = dio.read_emd(emd1)
testnode = "/Data/Image/6fdbde41eecc4375b45cd86bd2be17c0/Data"
testgroup = "/Data/Image/6fdbde41eecc4375b45cd86bd2be17c0"
testnode_ss = ("/Data/SpectrumStream/"
               "f5a4ba0965a5444b8c46cc420cf7fef0/Data")
testgroup_ss = ("/Data/SpectrumStream/"
                "f5a4ba0965a5444b8c46cc420cf7fef0")
ds = d[testnode]
ds_ss = d[testnode_ss]

emd2 = "examples/data/88_20_ SI 1511 15.7 Mx EDS-HAADF 20200307.emd"
d2 = dio.read_emd(emd2)


def test_read_emd():
    assert type(d) == dio.EMDFile


def test_json_functions():
    t = {"t": ["e", "s", {"t": 1, "e": "2"}], "s": {"t": 3, "e": "g"}}
    dio.write_to_json("./tests/test.json", t)
    tt = dio.read_json("./tests/test.json")
    assert t == tt
    srt = dio.get_pretty_dic_str(t)
    with open("./tests/test.json") as f:
        srtt = f.read()
    assert srtt == srt
    os.remove("./tests/test.json")


def test_EMDFile_get_ds_info():
    uuid, shape, dt = d._get_ds_info(ds)
    assert uuid == "6fdbde41eecc4375b45cd86bd2be17c0"
    assert shape == (256, 256, 240)
    assert str(dt) == "uint16"


def test_EMDFile_get_name():
    nmds = d._get_name(ds, full_path=False)
    assert nmds == "Data"
    nmds = d._get_name(ds, full_path=True)
    assert nmds == testnode
    nmgp = d._get_name(ds.parent, full_path=False)
    assert nmgp == "6fdbde41eecc4375b45cd86bd2be17c0"
    nmgp = d._get_name(ds.parent, full_path=True)
    assert nmgp == testgroup
    assert d._get_name(d, full_path=False) == "/"
    assert d._get_name(d, full_path=True) == "/"


def test_EMDFile_scan_node_print():
    d._scan_node(d["Data"], tabs=0, recursive=True,
                 see_info=True, tab_step=4)
    d._scan_node(d["Data/Image"], tabs=1, recursive=False,
                 see_info=False, tab_step=2)
    d.print_raw_structure()


def test_EMDFile_get_simple_rep():
    sim = d._get_simple_im_rep(testnode)
    assert sim == ("UUID: 6fdbde41eecc4375b45cd86bd2be17c0, "
                   "Shape: 256x256, "
                   "Frames: 240, "
                   "Data type: uint16, "
                   "Min:17381, "
                   "Max:57476")
    sss = d._get_simple_ss_rep(testnode_ss)
    assert sss == ("UUID: f5a4ba0965a5444b8c46cc420cf7fef0, "
                   "Length: 15933353, "
                   "Data type: uint16, ")
    d.get_simple_structure()


def test_EMDFile_get_ds_uuid():
    uuid = d._get_ds_uuid("Image", 0)
    assert uuid == "6fdbde41eecc4375b45cd86bd2be17c0"
    uuid = d._get_ds_uuid("Image", 2)
    assert uuid is None
    uuid = d._get_ds_uuid("Piet", 0)
    assert uuid is None
    uuid = d._get_ds_uuid("SpectrumStream", 0)
    assert uuid == "f5a4ba0965a5444b8c46cc420cf7fef0"


def test_EMDFile_get_meta_dict():
    meta = d._get_meta_dict("Image",
                            "6fdbde41eecc4375b45cd86bd2be17c0",
                            frame=0)
    assert type(meta) == dict
    meta = d._get_meta_dict_ds_no("Image", 0, frame=0)
    assert type(meta) == dict
    meta_ss = d._get_meta_dict("SpectrumStream",
                               "f5a4ba0965a5444b8c46cc420cf7fef0",
                               frame=0)
    assert type(meta_ss) == dict
    meta_ss = d._get_meta_dict_by_path(testgroup_ss)
    assert type(meta_ss) == dict


def test_EMDFile_get_detector_property():
    prop = d._get_detector_property(
        "Image",
        "6fdbde41eecc4375b45cd86bd2be17c0",
        "Dispersion",
        frame=0,
        exact_match=False)
    assert prop is None
    prop = d._get_detector_property(
        "SpectrumStream",
        "f5a4ba0965a5444b8c46cc420cf7fef0",
        "Dispersion",
        frame=0,
        exact_match=False)
    assert type(prop) == str


def test_EMDFile_get_scale():
    meta = d._get_meta_dict_ds_no("Image", 0, frame=0)
    psize, pscale = d._get_scale(meta)
    assert type(psize) == float
    assert type(pscale) == str
    meta_ss = d._get_meta_dict_by_path(testgroup_ss)
    psize, pscale = d._get_scale(meta_ss)
    assert type(psize) == float
    assert type(pscale) == str


def test_EMDFile_for_all_datasets_do():
    k = d._for_all_datasets_do(d._get_sig_ds_from_path)
    assert type(k) == list
    assert ("Image", "6fdbde41eecc4375b45cd86bd2be17c0") in k
    assert ("SpectrumStream", "f5a4ba0965a5444b8c46cc420cf7fef0") in k
    assert ("Text", "e1c99057161c4e9889c42955fab1e695") not in k


def test_EMDFile_get_spectrum_stream_acqset():
    t = d._get_spectrum_stream_acqset(
           "f5a4ba0965a5444b8c46cc420cf7fef0")
    assert type(t) == dict
    t = d._get_spectrum_stream_acqset(
           "6fdbde41eecc4375b45cd86bd2be17c0")
    assert t is None


def test_EMDFile_get_spectrum_stream_flut():
    flut = d._get_spectrum_stream_flut(
              "f5a4ba0965a5444b8c46cc420cf7fef0")
    assert flut is None
    uuid = d2._get_ds_uuid("SpectrumStream", 0)
    flut = d2._get_spectrum_stream_flut(uuid)
    assert type(flut) == np.ndarray


def test_EMDFile_get_spectrum_stream_dim():
    uuid1 = d._get_ds_uuid("SpectrumStream", 0)
    dim1 = d._get_spectrum_stream_dim(uuid1)
    assert dim1 == ()
    uuid2 = d2._get_ds_uuid("SpectrumStream", 0)
    dim2 = d._get_spectrum_stream_dim(uuid2)
    assert dim2 == ()


def test_EMDFile_convert_stream_to_sparse():
    pass
