module {
  func.func @main(%arg0: tensor<31x11x77xi8>, %arg1: tensor<31x77x2xi8>, %arg2: tensor<50x71x23x83x80xf32>, %arg3: tensor<1x71x23x83x1xf32>, %arg4: tensor<2x91x95x57x56x72xi1>) -> (tensor<2x91x95x57x56x72xi1>, tensor<50x71x23x83x80xf32>, tensor<50x71x23x83x80xf32>, tensor<93x33x6xi8>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<31x11x77xi8>, tensor<31x77x2xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<31x11x2xi8>
    %1 = tosa.bitwise_and %0, %0 : (tensor<31x11x2xi8>, tensor<31x11x2xi8>) -> tensor<31x11x2xi8>
    %2 = tosa.pow %arg2, %arg3 : (tensor<50x71x23x83x80xf32>, tensor<1x71x23x83x1xf32>) -> tensor<50x71x23x83x80xf32>
    %3 = tosa.logical_not %arg4 : (tensor<2x91x95x57x56x72xi1>) -> tensor<2x91x95x57x56x72xi1>
    %t_4 = tosa.const_shape {values = dense<[ 3, 3, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.tile %1, %t_4 : (tensor<31x11x2xi8>, !tosa.shape<3>) -> tensor<93x33x6xi8>
    %5 = tosa.add %2, %2 : (tensor<50x71x23x83x80xf32>, tensor<50x71x23x83x80xf32>) -> tensor<50x71x23x83x80xf32>
    %6 = tosa.clamp %4 {min_val = 4 : i8, max_val = 61 : i8} : (tensor<93x33x6xi8>) -> tensor<93x33x6xi8>
    %in_zp_7 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_7 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %7 = tosa.negate %2, %in_zp_7, %out_zp_7 : (tensor<50x71x23x83x80xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<50x71x23x83x80xf32>
    %8 = tosa.bitwise_not %6 : (tensor<93x33x6xi8>) -> tensor<93x33x6xi8>
    return %3, %5, %7, %8 : tensor<2x91x95x57x56x72xi1>, tensor<50x71x23x83x80xf32>, tensor<50x71x23x83x80xf32>, tensor<93x33x6xi8>
  }
}
