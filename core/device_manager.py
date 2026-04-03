try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def get_device_info() -> dict:
    """Return info about available compute device for OpenCV."""
    info = {"opencl_available": False, "device_name": "CPU (OpenCV)"}

    if not _CV2_AVAILABLE:
        info["device_name"] = "CPU (cv2 not installed)"
        return info

    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                device = cv2.ocl.Device.getDefault()
                info["opencl_available"] = True
                try:
                    info["device_name"] = f"OpenCL: {device.name()}"
                except Exception:
                    info["device_name"] = "OpenCL device"
            else:
                info["device_name"] = "CPU (OpenCL disabled)"
        else:
            info["device_name"] = "CPU (no OpenCL)"
    except Exception as exc:
        info["device_name"] = f"CPU (error: {exc})"

    return info
