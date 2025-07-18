module {
  func.func @main(%arg0: tensor<53x62x50x63x47xi32>, %arg1: tensor<53x1x50x63x47xi32>, %arg2: tensor<f32>, %arg3: tensor<63x52x76x15xi16>, %arg4: tensor<31x42xi1>) -> (tensor<47x63x62x50x53xi32>, tensor<53x62x50x63x47xi32>, tensor<53x62x50x63x47xi1>, tensor<31x42xi1>, tensor<f32>, tensor<63x52x76x1xi16>, tensor<31x1xi1>, tensor<i1>, tensor<63x52x76x1xi16>, tensor<6x6xi1>, tensor<63x52x76x1xi16>, tensor<31x1xi1>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<53x62x50x63x47xi32>, tensor<53x1x50x63x47xi32>) -> tensor<53x62x50x63x47xi32>
    %1 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<53x62x50x63x47xi32>) -> tensor<47x63x62x50x53xi32>
    %3 = tosa.rsqrt %arg2 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.sigmoid %3 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.ceil %3 : (tensor<f32>) -> tensor<f32>
    %6 = tosa.rsqrt %4 : (tensor<f32>) -> tensor<f32>
    %7 = tosa.bitwise_xor %0, %0 : (tensor<53x62x50x63x47xi32>, tensor<53x62x50x63x47xi32>) -> tensor<53x62x50x63x47xi32>
    %8 = tosa.rsqrt %6 : (tensor<f32>) -> tensor<f32>
    %9 = tosa.bitwise_or %7, %0 : (tensor<53x62x50x63x47xi32>, tensor<53x62x50x63x47xi32>) -> tensor<53x62x50x63x47xi32>
    %10 = tosa.exp %8 : (tensor<f32>) -> tensor<f32>
    %11 = tosa.tanh %5 : (tensor<f32>) -> tensor<f32>
    %12 = tosa.reduce_min %arg3 {axis = 3 : i32} : (tensor<63x52x76x15xi16>) -> tensor<63x52x76x1xi16>
    %13 = tosa.logical_not %arg4 : (tensor<31x42xi1>) -> tensor<31x42xi1>
    %14 = tosa.bitwise_xor %13, %13 : (tensor<31x42xi1>, tensor<31x42xi1>) -> tensor<31x42xi1>
    %15 = tosa.greater %10, %11 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %16 = tosa.greater_equal %0, %7 : (tensor<53x62x50x63x47xi32>, tensor<53x62x50x63x47xi32>) -> tensor<53x62x50x63x47xi1>
    %17 = tosa.logical_and %14, %14 : (tensor<31x42xi1>, tensor<31x42xi1>) -> tensor<31x42xi1>
    %18 = tosa.floor %10 : (tensor<f32>) -> tensor<f32>
    %in_zp_19 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_19 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %19 = tosa.negate %12, %in_zp_19, %out_zp_19 : (tensor<63x52x76x1xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<63x52x76x1xi16>
    %20 = tosa.sub %15, %15 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %21 = tosa.clz %19 : (tensor<63x52x76x1xi16>) -> tensor<63x52x76x1xi16>
    %22 = tosa.reduce_max %13 {axis = 1 : i32} : (tensor<31x42xi1>) -> tensor<31x1xi1>
    %23 = tosa.bitwise_or %20, %20 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %in_zp_24 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_24 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %24 = tosa.negate %19, %in_zp_24, %out_zp_24 : (tensor<63x52x76x1xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<63x52x76x1xi16>
    %s_25_start = tosa.const_shape {values = dense<[ 25, 20 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_25_size = tosa.const_shape {values = dense<[ 6, 6 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %25 = tosa.slice %13, %s_25_start, %s_25_size : (tensor<31x42xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<6x6xi1>
    %26 = tosa.reduce_sum %19 {axis = 3 : i32} : (tensor<63x52x76x1xi16>) -> tensor<63x52x76x1xi16>
    %27 = tosa.reduce_any %13 {axis = 1 : i32} : (tensor<31x42xi1>) -> tensor<31x1xi1>
    return %2, %9, %16, %17, %18, %21, %22, %23, %24, %25, %26, %27 : tensor<47x63x62x50x53xi32>, tensor<53x62x50x63x47xi32>, tensor<53x62x50x63x47xi1>, tensor<31x42xi1>, tensor<f32>, tensor<63x52x76x1xi16>, tensor<31x1xi1>, tensor<i1>, tensor<63x52x76x1xi16>, tensor<6x6xi1>, tensor<63x52x76x1xi16>, tensor<31x1xi1>
  }
}
