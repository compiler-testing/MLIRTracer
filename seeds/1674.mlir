module {
  func.func @main(%arg0: tensor<19x69x99xi1>, %arg1: tensor<41x53x16x57x64x58xf32>) -> (tensor<2x69x99xi1>, tensor<41x53x16x57x64x58xf32>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<19x69x99xi1>) -> tensor<1x69x99xi1>
    %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<1x69x99xi1>, tensor<1x69x99xi1>) -> tensor<2x69x99xi1>
    %2 = tosa.reverse %1 {axis = 2 : i32} : (tensor<2x69x99xi1>) -> tensor<2x69x99xi1>
    %3 = tosa.arithmetic_right_shift %2, %1 {round = false} : (tensor<2x69x99xi1>, tensor<2x69x99xi1>) -> tensor<2x69x99xi1>
    %4 = tosa.reciprocal %arg1 : (tensor<41x53x16x57x64x58xf32>) -> tensor<41x53x16x57x64x58xf32>
    return %3, %4 : tensor<2x69x99xi1>, tensor<41x53x16x57x64x58xf32>
  }
}
