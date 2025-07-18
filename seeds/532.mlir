module {
  func.func @main(%arg0: tensor<37x47x1x48xi8>, %arg1: tensor<1x1x1x48xi8>, %arg2: tensor<53x95x46x74x65x57xf32>, %arg3: tensor<24x26xi1>) -> (tensor<37x47x1x48xi8>, tensor<53x95x46x74x65x57xf32>, tensor<37x47x1x48xi8>, tensor<53x95x46x74x65x57xf32>, tensor<53x95x46x74x65x57xf32>, tensor<24x1xi1>, tensor<53x95x46x74x65x57xf32>, tensor<6x8x7x5x9x10xf32>, tensor<37x47x1x48xi8>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<37x47x1x48xi8>, tensor<1x1x1x48xi8>) -> tensor<37x47x1x48xi8>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<37x47x1x48xi8>, tensor<37x47x1x48xi8>) -> tensor<37x47x1x48xi8>
    %2 = tosa.log %arg2 : (tensor<53x95x46x74x65x57xf32>) -> tensor<53x95x46x74x65x57xf32>
    %3 = tosa.logical_left_shift %1, %1 : (tensor<37x47x1x48xi8>, tensor<37x47x1x48xi8>) -> tensor<37x47x1x48xi8>
    %4 = tosa.log %2 : (tensor<53x95x46x74x65x57xf32>) -> tensor<53x95x46x74x65x57xf32>
    %5 = tosa.arithmetic_right_shift %1, %0 {round = true} : (tensor<37x47x1x48xi8>, tensor<37x47x1x48xi8>) -> tensor<37x47x1x48xi8>
    %6 = tosa.reduce_any %arg3 {axis = 1 : i32} : (tensor<24x26xi1>) -> tensor<24x1xi1>
    %7 = tosa.abs %2 : (tensor<53x95x46x74x65x57xf32>) -> tensor<53x95x46x74x65x57xf32>
    %8 = tosa.logical_right_shift %5, %5 : (tensor<37x47x1x48xi8>, tensor<37x47x1x48xi8>) -> tensor<37x47x1x48xi8>
    %9 = tosa.logical_left_shift %1, %0 : (tensor<37x47x1x48xi8>, tensor<37x47x1x48xi8>) -> tensor<37x47x1x48xi8>
    %10 = tosa.sub %2, %4 : (tensor<53x95x46x74x65x57xf32>, tensor<53x95x46x74x65x57xf32>) -> tensor<53x95x46x74x65x57xf32>
    %11 = tosa.maximum %4, %4 : (tensor<53x95x46x74x65x57xf32>, tensor<53x95x46x74x65x57xf32>) -> tensor<53x95x46x74x65x57xf32>
    %12 = tosa.pow %2, %2 : (tensor<53x95x46x74x65x57xf32>, tensor<53x95x46x74x65x57xf32>) -> tensor<53x95x46x74x65x57xf32>
    %13 = tosa.reduce_product %6 {axis = 1 : i32} : (tensor<24x1xi1>) -> tensor<24x1xi1>
    %14 = tosa.sigmoid %11 : (tensor<53x95x46x74x65x57xf32>) -> tensor<53x95x46x74x65x57xf32>
    %s_15_start = tosa.const_shape {values = dense<[ 46, 49, 12, 29, 49, 1 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_15_size = tosa.const_shape {values = dense<[ 6, 8, 7, 5, 9, 10 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %15 = tosa.slice %2, %s_15_start, %s_15_size : (tensor<53x95x46x74x65x57xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<6x8x7x5x9x10xf32>
    %16 = tosa.clz %8 : (tensor<37x47x1x48xi8>) -> tensor<37x47x1x48xi8>
    return %3, %7, %9, %10, %12, %13, %14, %15, %16 : tensor<37x47x1x48xi8>, tensor<53x95x46x74x65x57xf32>, tensor<37x47x1x48xi8>, tensor<53x95x46x74x65x57xf32>, tensor<53x95x46x74x65x57xf32>, tensor<24x1xi1>, tensor<53x95x46x74x65x57xf32>, tensor<6x8x7x5x9x10xf32>, tensor<37x47x1x48xi8>
  }
}
