module {
  func.func @main(%arg0: tensor<34x12x37x13xf32>, %arg1: tensor<6x52x47xi1>, %arg2: tensor<1x1x1xi1>) -> (tensor<12x104x47xi1>, tensor<12x156x141xi1>, tensor<34x12x37x13xf32>) {
    %0 = tosa.ceil %arg0 : (tensor<34x12x37x13xf32>) -> tensor<34x12x37x13xf32>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<6x52x47xi1>, tensor<1x1x1xi1>) -> tensor<6x52x47xi1>
    %2 = tosa.logical_not %1 : (tensor<6x52x47xi1>) -> tensor<6x52x47xi1>
    %3 = tosa.rsqrt %0 : (tensor<34x12x37x13xf32>) -> tensor<34x12x37x13xf32>
    %4 = tosa.logical_or %1, %2 : (tensor<6x52x47xi1>, tensor<6x52x47xi1>) -> tensor<6x52x47xi1>
    %t_5 = tosa.const_shape {values = dense<[ 2, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.tile %4, %t_5 : (tensor<6x52x47xi1>, !tosa.shape<3>) -> tensor<12x104x47xi1>
    %t_6 = tosa.const_shape {values = dense<[ 2, 3, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %6 = tosa.tile %2, %t_6 : (tensor<6x52x47xi1>, !tosa.shape<3>) -> tensor<12x156x141xi1>
    %7 = tosa.maximum %0, %3 : (tensor<34x12x37x13xf32>, tensor<34x12x37x13xf32>) -> tensor<34x12x37x13xf32>
    return %5, %6, %7 : tensor<12x104x47xi1>, tensor<12x156x141xi1>, tensor<34x12x37x13xf32>
  }
}
