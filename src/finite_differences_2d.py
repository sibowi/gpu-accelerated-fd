import torch


def dx_centered(array):
    """
    Computes the centered finite difference of the inputted array along the x
    direction. Ie. across rows.

    Parameters
    ----------
    array : np.ndarray
        The array of interest.

    Returns
    -------
    dx : np.ndarray
        The derivative of array.
    """

    step_size = 2

    dx = (array[:,step_size:] - array[:,:-step_size])/(step_size)

    dx = _pad(dx, mode = 'edge', dim = 1)

    return dx



def dy_centered(array,y_step=None,step_size=None):
    """
    Computes the centered finite difference of the inputted array along the y
    direction. Ie. across columns.

    Parameters
    ----------
    array : torch.Tensor
        The array of interest.

    Returns
    -------
    dy : torch.Tensor
        The centered first-order derivative of array.
    """

    step_size = 2

    dy = (array[step_size:,:] - array[:-step_size,:])/(step_size)

    dy = _pad(dy, mode = 'edge', dim = 0)

    return dy



def dxdx_centered(array):
    """
    Computes the centered second-order finite difference of the inputted array
    along the x direction. Ie. across rows.
    
    Parameters
    ----------
    array : torch.Tensor
        The array of interest.

    Returns
    -------
    dxdx : torch.Tensor
        The centered second-order derivative of array.
    """

    dxdx = (array[:,2:] - 2*array[:,1:-1] + array[:,:-2])

    dxdx = _pad(dxdx, mode = 'edge', dim = 1)

    return dxdx



def dydy_centered(array):
    """
    Computes the centered second-order finite difference of the inputted array
    along the y direction. Ie. across columns.

    Parameters
    ----------
    array : torch.Tensor
        The array of interest.

    Returns
    -------
    dydy : torch.Tensor
        The centered second-order derivative of array.    
    """

    dydy = (array[2:, :] - 2*array[1:-1, :] + array[:-2, :])

    dydy = _pad(dydy, mode = 'edge', dim = 0)

    return dydy



def _pad(array, mode=None, dim=None):
    # because "Only 3D, 4D, 5D padding with non-constant padding are supported"
    # for torch.nn.functional.pad() for now
    # TODO: Update when available.

    """
    Performs padding of a 2d tensor.

    Parameters
    ----------
    array : torch.Tensor
        The tensor which will be padded.
    mode : str
        The mode that will be used for handling boundaries.
    dim : int
        The dimension in which the padding is performed.

    Returns
    -------
    array : torch.Tensor
        The padded tensor.
    """

    assert mode != None, 'mode was not defined.'
    assert dim != None, 'dim was not defined'

    if mode == 'edge':
        if dim == 0:
            r_f = array[:1,:] #first row
            r_l = array[-1:,:] #last row
            array = torch.cat((r_f,array,r_l), dim=dim)
        if dim == 1:
            c_f = array[:,:1] #first column
            c_l = array[:,-1:] #last column
            array = torch.cat((c_f,array,c_l), dim=dim)
    else:
        print('mode is unknown.')

    return array
