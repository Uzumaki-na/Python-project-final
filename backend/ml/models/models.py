import timm

def SkinCancerModel():
    """Returns a timm efficientnet_b0 model for skin cancer classification"""
    return timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)

def MalariaModel():
    """Returns a timm efficientnet_lite0 model for malaria classification"""
    return timm.create_model('efficientnet_lite0', pretrained=False, num_classes=2)
