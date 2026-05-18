#include "pogobot.h"
#include "time.h"
#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INFRARED_POWER 2 // 1,2,3

#define FQCY 60 // control update frequency. 30Hz | 60 Hz | 90 Hz | etc.
#define MAX_NB_OF_MSG 3 // max. number of messages per step which this robot can record // 3

#define SEND_MODE_ALLDIRECTION true // true: all direction at once; false: 4x one-direction
#define MSG_MODE_FULL_HEADER true // true: full header; false: short header

#define DEBUG_LEVEL 0 // 0: nothing; 1: debug; 2: synchronzation; 3: communication

#define BOOT_TIME 5 // waiting time before the start of the experience in secondsfor(int i=0;i<5;i++) pogobot_led_setColors(0,0,0,i);
#define LIGHT_THRESHOLD 10

#define MOTOR_POWER 718 // default power of the motor in the else case
#define IDENTIFIER 1
#define P_MSG 1
#define MESSAGE 1 // 1 : enable communication reception     
#define DEFAULT_MODE MODE_NORMAL // Default mode of the robot when not root

#define MAX_GRID 2 // Grille de Taille Max_Grid + 1 
#define PWR_SLOW 380


#define FLAG_BOOT_TIME 10
#define RESET_NB_IT 100



// ********************************************************************************
// * FLAGGGGG
// * 
// ********************************************************************************


// ********************************************************************************
// * Controller 1
// * Insert here the Pogobot controller (in flagAutomata > src > simulationAnalysis > ... > learning > run_00X > data > data_env_controllers)
// ********************************************************************************

// flagAutomata individual controller run 005, gen 01137, eval 0013650, nb_ind 005
uint16_t ann1_nb_layers = 1;  // nb_layers = 1 input layer + N hidden layers. Output layer excluded
uint16_t ann1_input_size = 4;
uint16_t ann1_output_size = 3;
uint16_t ann1_nb_neurons_l0 = 4;
uint16_t ann1_nb_neurons_l1 = 3;
const double ann1_weights_l0_l1[4][3] = {{1.657529087536642, -5.693473362807253e-10, -14.360303863493826}, {0.36948342338722107, -16.28174309395963, -1.3883482434450795e-09}, {8.691410612524114, 11.18618053661279, -9.518330154726985e-10}, {5.5109002178463875, -2.064617276422583e-09, 10.013473833401028}};
const double ann1_biases_l0_l1[3] = {-10.768765433231158, -5.095562560351923, -4.346830032059591};
const void* ann1_weights[] = {ann1_weights_l0_l1};
const double* ann1_biases[] = {ann1_biases_l0_l1};
const int ann1_weights_sizes[][2] = {{4, 3}};
const int ann1_biases_sizes[] = {3};
double ann1_input_layer[4] = {3};
double ann1_output_layer[3] = {3};

#define ANN1_MAX_NB_NEURONS_ALL_LAYERS 4

#define ANN1_ACTIVATION_FUNCTION_ARRAY tanh_activation_array

#define ANN1_AGENT "agentCoordinates_gradient"

#define ANN1_CHEMICALS_TO_SPREAD_SIZE 1

#define ANN1_PHENOTYPE_SIZE 2

#define ANN1_STACKING_MODE "None"

// flagAutomata individual controller run 014, gen 00351, eval 0005981, nb_ind 013
uint16_t ann2_nb_layers = 3;  // nb_layers = 1 input layer + N hidden layers. Output layer excluded
uint16_t ann2_input_size = 6;
uint16_t ann2_output_size = 3;
uint16_t ann2_nb_neurons_l0 = 6;
uint16_t ann2_nb_neurons_l1 = 5;
uint16_t ann2_nb_neurons_l2 = 5;
uint16_t ann2_nb_neurons_l3 = 3;
const double ann2_weights_l0_l1[6][5] = {{0.7552518945370189, 0.2274221287230821, 6.399259528616099, -1.02310900566743, -6.54178742254187}, {4.6633132811304465, -7.0916329555779996, 5.413062768786927, 8.69799482952632, -1.0449246812263098}, {4.798043928090804, -1.3365549239731167, -11.121422549342288, -5.417828448167219, -0.8868252303197601}, {2.317271045491729, -5.874679267240691, -0.8358172288804825, 2.3796676046940726, -10.052717116834598}, {0.9512184303874388, -3.467317074912458, -11.672206393492075, 3.3827643465682313, -2.938058788430701}, {3.078381238658499, -2.739724522767221, 2.3477708479484782, -0.08741626408158995, -1.5611497030736339}};
const double ann2_weights_l1_l2[5][5] = {{7.388528714753382, 5.935767285041721, 4.74786672157737, 1.4502756421182426, -11.098646607069538}, {-5.971253720054059, 1.2548943476546626, 5.864993320689067, -1.81214663285053, 0.33696248323204514}, {-2.7144351027508535, -0.6883394903477187, 0.7919277397269591, -3.3748564079031063, 1.6701512134696883}, {3.239513690825322, -3.3271054022025512, -1.5351424841309966, -3.743524498004757, 8.411341253850813}, {-1.422822578500345, -5.7543487990733, -5.41298897962362, -4.781188440401226, 7.262823386891325}};
const double ann2_weights_l2_l3[5][3] = {{0.3426988323377387, -0.35292208584525986, 1.2441928886873295}, {13.00666071418499, 14.028966087434076, -1.8672198846206856}, {9.087844357215914, 5.850159355292347, -0.18305591699987223}, {-3.3237382302766045, -4.890788581651278, -11.309395520400189}, {-4.424699799721179, 15.864071076906042, 12.56989523076498}};
const double ann2_biases_l0_l1[5] = {5.887428980155922, -1.32769093111808, 6.099689453934921, 0.29006527778690594, -8.726322724446703};
const double ann2_biases_l1_l2[5] = {5.547185954614734, 10.314797023391156, 2.527120411256689, 2.324291365530048, 1.9705460518036393};
const double ann2_biases_l2_l3[3] = {1.9962793257671758, -20.763005781107655, 2.287540366360439};
const void* ann2_weights[] = {ann2_weights_l0_l1, ann2_weights_l1_l2, ann2_weights_l2_l3};
const double* ann2_biases[] = {ann2_biases_l0_l1, ann2_biases_l1_l2, ann2_biases_l2_l3};
const int ann2_weights_sizes[][2] = {{6, 5}, {5, 5}, {5, 3}};
const int ann2_biases_sizes[] = {5, 5, 3};
double ann2_input_layer[6] = {5, 5, 3};
double ann2_output_layer[3] = {5, 5, 3};

#define ANN2_MAX_NB_NEURONS_ALL_LAYERS 6

#define ANN2_ACTIVATION_FUNCTION_ARRAY tanh_activation_array

#define ANN2_AGENT "agent2Outputs_RGB"

#define ANN2_CHEMICALS_TO_SPREAD_SIZE 1

#define ANN2_PHENOTYPE_SIZE 3

#define ANN2_STACKING_MODE "ann1_ann2_modelC"


// ********************************************************************************
// * Other global variables
// ********************************************************************************

#ifdef ANN2_MAX_NB_NEURONS_ALL_LAYERS
    #define MAX(a,b) ((a)>(b)?(a):(b))
    #define MAX_NB_NEURONS_ALL_LAYERS MAX(ANN1_MAX_NB_NEURONS_ALL_LAYERS, ANN2_MAX_NB_NEURONS_ALL_LAYERS)
    #define CHEMICALS_TO_SPREAD_SIZE ANN2_CHEMICALS_TO_SPREAD_SIZE
    #define PHENOTYPE_SIZE ANN2_PHENOTYPE_SIZE
    #define NB_AGENT_CONTROLLERS 2
#else
    #define MAX_NB_NEURONS_ALL_LAYERS ANN1_MAX_NB_NEURONS_ALL_LAYERS
    #define CHEMICALS_TO_SPREAD_SIZE ANN1_CHEMICALS_TO_SPREAD_SIZE
    #define PHENOTYPE_SIZE ANN1_PHENOTYPE_SIZE
    #define NB_AGENT_CONTROLLERS 1
#endif

double previous_layer[MAX_NB_NEURONS_ALL_LAYERS] = {0.0};
double weighted_sum[MAX_NB_NEURONS_ALL_LAYERS] = {0.0};
double my_chemicals_to_spread[CHEMICALS_TO_SPREAD_SIZE];
double my_phenotype[PHENOTYPE_SIZE];
int my_led_index = -1;
int color_index = -1;


// ********************************************************************************
// * Pogobot message structure
// ********************************************************************************

typedef struct RawMessage {
    uint16_t sender_id;
    uint16_t age;
    double chemicals_to_spread[CHEMICALS_TO_SPREAD_SIZE];
} RawMessage;

#define MSG_SIZE sizeof(RawMessage) // number of bytes

typedef union message_template {
    uint8_t msg_array[MSG_SIZE];
    RawMessage msg_values;
} message;

// ********************************************************************************
// * Structure pour les messages
// ********************************************************************************

typedef struct grid_msg_t {                               // État de l'émetteur (1 = actif). Utile pour filtrer.
    uint8_t row;   // 0..255                                        // Ligne annoncée par le voisin (on revalidera la borne).
    uint8_t col;   // 0..255                                        // Colonne annoncée par le voisin.
} grid_msg_t;
// Note : "typedef struct {...} grid_msg_t;" crée un alias "grid_msg_t". Sans typedef, il faudrait écrire "struct grid_msg_t".
#define GRID_MSG_SIZE ((uint16_t)sizeof(grid_msg_t))                // Taille en octets d'un message (utile pour memcpy/envoi).

grid_msg_t pos;
// ********************************************************************************
// * Initialization: Pogobot led colors structure
// ********************************************************************************

typedef struct {
    char name[16];
    uint8_t r;
    uint8_t g;
    uint8_t b;
} rgb_color; 

rgb_color red =         {.name = "red",        .r = 25, .g = 0,  .b = 0};
rgb_color green =       {.name = "green",      .r = 0,  .g = 25, .b = 0};
rgb_color blue =        {.name = "blue",       .r = 0,  .g = 0,  .b = 25};
rgb_color magenta =     {.name = "magenta",    .r = 25, .g = 0,  .b = 25};
rgb_color yellow =      {.name = "yellow",     .r = 12, .g = 6,  .b = 0};
rgb_color cyan =        {.name = "cyan",       .r = 0,  .g = 25, .b = 25};
rgb_color orange =      {.name = "orange",     .r = 25, .g = 6,  .b = 0};
rgb_color purple =      {.name = "purple",     .r = 6,  .g = 0,  .b = 25};
rgb_color light_pink =  {.name = "light_pink", .r = 12, .g = 3,  .b = 12};
rgb_color mint_green =  {.name = "mint_green", .r = 6,  .g = 25, .b = 6};
rgb_color white =       {.name = "white",      .r = 25, .g = 25, .b = 25};

#define RT_RUN_MIN_MS             1500                             // RUN = se déplacer en ligne droite (ou arrière) pendant 1.5 à 3 s.
#define RT_RUN_MAX_MS             3000
#define RT_TUMBLE_MIN_MS           400                             // TUMBLE = tourner sur place pendant 0.4 à 0.6 s.
#define RT_TUMBLE_MAX_MS           600
#define A_TTL_MS 3000
#define CHECK_PHASE_MS 100
#define CON_TIMER_MS 3000
#define CHECK_TIMER_MS 1600
#define REC_INPUT_TIMER_MS 300
#define SEEK_MS 1000
#define FINISH_MS 3000
#define MAX_MS 6000
#define FACT 1.0
#define FACT1 1.0

bool flag_light_initialized = false;

typedef enum { 
    MODE_NORMAL, 
    MODE_ALIGN,
    MODE_CONNECTION_ROW,
    MODE_CONNECTION_COL,
    MODE_CHECK_CONNECTION_COL,
    MODE_CHECK_CONNECTION_ROW,
    MODE_CONNECTED,
    MODE_FLAG,
    MODE_ALIGN_TO_SEEK,
    MODE_SEEK_ROOT,
    MODE_ROOT,
    MODE_FLAG_LIGHT
} mobile_mode_t;

// Sous-phases pour le run&tumble (utile pour alterner).
typedef enum { RT_PHASE_RUN=0, RT_PHASE_TUMBLE } rt_phase_t;
typedef enum { CHECK_PHASE_FW, CHECK_PHASE_TURN} check_phase_t;

// ===================== État global =====================

uint8_t dirL, dirR;
uint32_t pwmLt, pwmRt, pwmLr, pwmRr;

mobile_mode_t mode;        
rt_phase_t rt_phase;
time_reference_t rt_phase_timer;


check_phase_t check_phase;
time_reference_t check_phase_timer;
time_reference_t con_timer;
time_reference_t check_timer;
time_reference_t rec_input_timer;
time_reference_t seek_timer; 
time_reference_t a_timer; 
time_reference_t max_timer;
time_reference_t finish_timer;  
uint32_t a_ttl;
uint32_t check_ttl;
uint32_t con_ttl;
uint32_t check_phase_ttl;
uint32_t rec_input_ttl;
uint32_t seek_ttl;
uint32_t finish_ttl;
uint32_t max_ttl;

uint32_t rt_duration_ms;   
bool rt_tumble_left;
bool rt_run_backward;  

int robot_identifier  = IDENTIFIER;

    // ********************************************************************************
    // * Initialization: parameters and initial conditions
    // ********************************************************************************



time_reference_t timeout_N = {0};
time_reference_t timeout_W = {0};
time_reference_t timeout_E = {0};
time_reference_t timeout_S = {0};
time_reference_t timeout_NWES[4];

uint32_t seconds = 0;
uint32_t max_chemicals_to_spread_validity_time_seconds = 30;

time_reference_t timeout_phenotypic_freshness;
uint32_t phenotypic_freshness_time_seconds = 3;
uint32_t hanabi_blinking_time_seconds = 0.8;

uint16_t my_pogobot_id = 0;
uint16_t age = 0;

double p_send_per_step = 1.0/2.0;
uint16_t counter_rcvd_msgs = 0;
uint8_t data[MSG_SIZE];

message msg_from_neighbor;

uint8_t nb_rgb_colors = 7;
double p_change_led_color = 1.0/1000.0;

uint16_t nb_ir = 0;
uint8_t my_receiver_ir_index = 5;
uint8_t starting_index_NWES = 5;

// ********************************************************************************
// * Functions
// ********************************************************************************

double tanh_approx(double x); // chatGPT
double sigmoid_activation_value(double value);
void tanh_activation_array(double* neurons_layer, int neurons_layer_size);
void sigmoid_activation_array(double* neurons_layer, int neurons_layer_size);
void fnn_predict(double *ann_input_layer, double *ann_output_layer, int nb_layers, const int (*weights_sizes)[2], const void **weights, const double **biases, void (*activation_function)(double*, int), int input_size, int output_size);
void display_phenotype(double* my_phenotype);
void update_robot_state(uint16_t age); // state = my_chemicals_to_spread + my_phenotype
void print_all_chemicals_infos(uint16_t my_pogobot_id, uint16_t age);

// parameters to print values of type "double" correctly
#define PRECISION 1000000 // conversion double<->int (ex.: 100 is 2 decimals) -- note: when sending/receiving, payload is a list of *bytes*
#define NB_DIGIT log_10(PRECISION)
int log_10(int power_of_ten);
int double_get_decimal_part(double value);
void double_get_decimal_part_string(double value, char *res);
int double_get_integer_part(double value);
void print_double(double val);

//-------------------------------------------------------------

double tanh_approx(double x) { // chatGPT
    if (x > 20.0) return 1.0;  // Large positive values return 1
    if (x < -20.0) return -1.0; // Large negative values return -1
    double exp_pos = exp(x);
    double exp_neg = exp(-x);
    return (exp_pos - exp_neg) / (exp_pos + exp_neg);
}

//-------------------------------------------------------------

void tanh_activation_array(double* neurons_layer, int neurons_layer_size){
    for (int i=0; i<neurons_layer_size; i++){
        neurons_layer[i] = tanh(neurons_layer[i]); // hyperbolic tangent
    }
}

//-------------------------------------------------------------

double sigmoid_activation_value(double value){ // sigmoid(x) = 1 / (1 + np.exp(-x))
    return 1 / (1 + exp(-value));
}

//-------------------------------------------------------------

void sigmoid_activation_array(double* neurons_layer, int neurons_layer_size){ // sigmoid(x) = 1 / (1 + np.exp(-x))
    int i;
    for (i=0; i<neurons_layer_size; i++){
        neurons_layer[i] = 1 / (1 + exp(-neurons_layer[i]));
    }
}

//-------------------------------------------------------------

void fnn_predict(
    double *ann_input_layer,
    double *ann_output_layer,
    int nb_layers,
    const int (*weights_sizes)[2],
    const void **weights,
    const double **biases,
    void (*activation_function)(double*, int),
    int input_size,
    int output_size){

    int nb_layer;
    int nb_neurons_prev_layer;
    int nb_neurons_next_layer;
    int n_next;
    int n_prev;
    int i;

    for(int i=0; i<input_size; i++){
        previous_layer[i] = ann_input_layer[i];
    }

    if (DEBUG_LEVEL == 5 || DEBUG_LEVEL == 99) {printf("\n[FNN] Updating chemicals. Forward Neural Network computation detail\n\tann_input_layer = [ "); for (int i = 0; i < ann1_input_size; i++) {print_double(previous_layer[i]); printf(" ");} printf("]\n");}

    for(nb_layer=0; nb_layer<nb_layers; nb_layer++){
        nb_neurons_prev_layer = weights_sizes[nb_layer][0];
        nb_neurons_next_layer = weights_sizes[nb_layer][1];

        // printf("\nnb_layer %d", nb_layer);
        // printf("\nb_neurons_prev_layer %d", nb_neurons_prev_layer);
        // printf("\nb_neurons_next_layer %d", nb_neurons_next_layer);

        if (DEBUG_LEVEL == 5 || DEBUG_LEVEL == 99) {printf("\n[FNN] ann_input_layer (previous layer) = [ "); for (i = 0; i < nb_neurons_prev_layer; i++) {print_double(previous_layer[i]); printf(" ");} printf("]\n");}

        for (n_next=0; n_next<nb_neurons_next_layer; n_next++){
            weighted_sum[n_next] = 0.0;
            //printf("nb neuron next layer %d\n", n_next);
            double (*weights_lprev_lnext)[nb_neurons_next_layer] = (double (*)[nb_neurons_next_layer]) weights[nb_layer];
            for (n_prev=0; n_prev<nb_neurons_prev_layer; n_prev++){
                //print_double(previous_layer[n_prev]); printf(" * \n");
                //print_double(weights_lprev_lnext[n_prev][n_next]); printf(" = \n");
                //print_double(weighted_sum[n_next]); printf(" --> ");
                weighted_sum[n_next] += previous_layer[n_prev] * weights_lprev_lnext[n_prev][n_next];
                //print_double(weighted_sum[n_next]); printf("\n");
            }
            //printf(" + bias (");
            weighted_sum[n_next] += biases[nb_layer][n_next]; //print_double(biases[nb_layer][n_next]); printf(" ) = "); print_double(weighted_sum[n_next]); printf("\n---\n");
        }
        if (DEBUG_LEVEL == 5 || DEBUG_LEVEL == 99) {printf("\tweighted_sum before activation layer %d = [ ", nb_layer); for (int i = 0; i < nb_neurons_next_layer; i++) {print_double(weighted_sum[i]); printf(" ");} printf("]\n");}
        activation_function(weighted_sum, nb_neurons_next_layer);
        for(i=0; i<nb_neurons_next_layer; i++){
            previous_layer[i] = weighted_sum[i]; // the last previous_layer is the output layer of the NN
        }
        
        if (DEBUG_LEVEL == 5 || DEBUG_LEVEL == 99) {printf("\tweighted_sum after activation layer %d = [ ", nb_layer); for (int i = 0; i < nb_neurons_next_layer; i++) {print_double(previous_layer[i]); printf(" ");} printf("]\n");}
    }

    for(i=0; i<output_size; i++){
        ann_output_layer[i] = previous_layer[i]; // the last previous_layer is the output layer of the NN
    }
    if (DEBUG_LEVEL == 5 || DEBUG_LEVEL == 99) {printf("\tann_output_layer_NWES = [ "); for (int i = 0; i < output_size; i++) {print_double(ann_output_layer[i]);} printf(" "); printf("]\n");}
}

//-------------------------------------------------------------

void display_phenotype(double* my_phenotype){

    switch (PHENOTYPE_SIZE){
        case 1:
            // Monochromatic flags, phenotype ranges from red (0.0) to blue (1.0)
            pogobot_led_setColor((uint8_t)((1.0 - my_phenotype[0])*25), (uint8_t) 0.0, (uint8_t)(my_phenotype[0]*25));
            break;
        case 3:
            // RGB flags, phenotype ranges from black (0.0, 0.0, 0.0) to white (1.0, 1.0, 1.0)
            pogobot_led_setColor((uint8_t)(my_phenotype[0]*25), (uint8_t)(my_phenotype[1]*25), (uint8_t)(my_phenotype[2]*25));
            break;
    }
}

//-------------------------------------------------------------

void update_robot_state(uint16_t age){ // state = my_chemicals_to_spread + my_phenotype

    uint16_t i;

    // Reset of initial condition, to avoid the saturation of the inputs in case a high signals spreads (the ANN is not trained to manage different initial conditions)
    if (age%RESET_NB_IT == 0){
        for (i=0; i<ann1_input_size; i++){
            ann1_input_layer[i] = 0.0;
        }
        if (DEBUG_LEVEL == 3 || DEBUG_LEVEL == 99) { printf("\n[INPUT FNN] Reset of all neighbors external chemicals.\n"); for (int i = 0; i < ann1_input_size; i++) {printf("\tann1_input_layer[%d] = ", i); print_double(ann1_input_layer[i]); printf("\n");}}
        pogobot_led_setColor(blue.r, blue.g, blue.b); // blue reset blinking
        sleep(FLAG_BOOT_TIME); // giving time to reset the neighborhood
        pogobot_infrared_clear_message_queue();
    }
    else{
        pogobot_led_setColor(white.r, white.g, white.b); // white hanabi blinking
    }

    fnn_predict( // get the updated ANN1 ann_output_layer, containing chemicals and phenotypes (to normalize, eventually)
        ann1_input_layer,
        ann1_output_layer,
        ann1_nb_layers,
        ann1_weights_sizes,
        ann1_weights,
        ann1_biases,
        ANN1_ACTIVATION_FUNCTION_ARRAY,
        ann1_input_size,
        ann1_output_size
    );

    if (NB_AGENT_CONTROLLERS == 1){
        for (i=0; i<CHEMICALS_TO_SPREAD_SIZE; i++){
            my_chemicals_to_spread[i] = ann1_output_layer[i];
        }
        for (i=0; i<PHENOTYPE_SIZE; i++){
            my_phenotype[i] = (ann1_output_layer[ann1_output_size - PHENOTYPE_SIZE + i] + 1.0) / 2.0; // rescale phenotype x in (-1,1) to (0,1);
        }
        return;
    }

    // In the following options, we combine more than one ANN
    if (strcmp(ANN2_STACKING_MODE, "ann1_ann2_modelA") == 0){
        
        // Model A: 4-x-3_2-y-1
        // The 1st ANN (4-x-3), used for the learning phase (coordinates system, flag 2D), has:
        //   - inputs: a signal (chemicals_to_spread) from each neighbor. ann1 input = neighbors_states = [ signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ]
        //   - output: a signal to spread to neighbors, and two phenotype values x and y. ann1 output = [ signal_xy, x, y ]
        // The 2nd ANN (2-y-1), used to learn the target flag (two-bands, centered-half-discs, ...), has:
        //   - inputs: x and y from the coordinate system. ann2 input = [ x, y ]
        //   - output: one phenotype. ann2 output = [ p ]
        // The final state for an agent, is [signal_xy from ann1, p from ann2]
        // NB: all phenotypes (x, y, p) are rescaled from (-1,1) to (0,1); ann2 phenotypes (p) will be rescaled in agent2Outputs

        if (!( (strcmp(ANN1_AGENT, "agentCoordinates_gradient") == 0) && ( (strcmp(ANN2_AGENT, "agent2Outputs") == 0) || (strcmp(ANN2_AGENT, "agent2Outputs_RGB") == 0) ) )){
            printf("Error in compute_robot_state ann1_ann2_modelPIX-A: ANN2_AGENT not correct, check controller parameters.");
            exit(1);
        }

        // Here we prepare the ann2_input_layer
        ann2_input_layer[0] = (ann1_output_layer[1] + 1.0) / 2.0; // rescale phenotype x in (-1,1) to (0,1)
        ann2_input_layer[1] = (ann1_output_layer[2] + 1.0) / 2.0; // rescale phenotype y in (-1,1) to (0,1)

        fnn_predict(
            ann2_input_layer,  // we skip signal_xy and we send the pointer on x, as [x,y] are the ANN2 entries
            ann2_output_layer,
            ann2_nb_layers,
            ann2_weights_sizes,
            ann2_weights,
            ann2_biases,
            ANN2_ACTIVATION_FUNCTION_ARRAY,
            ann2_input_size,
            ann2_output_size
        );

        for (i=0; i<CHEMICALS_TO_SPREAD_SIZE; i++){
            my_chemicals_to_spread[i] = ann1_output_layer[i]; // ANN1 returns the chemicals to share with my neighbors via msg, for all setups
        }
        for (i=0; i<PHENOTYPE_SIZE; i++){
            my_phenotype[i] = (ann2_output_layer[ann2_output_size - PHENOTYPE_SIZE + i] + 1.0) / 2.0; // rescale phenotype x in (-1,1) to (0,1);
        }
    }

    else if (strcmp(ANN2_STACKING_MODE, "ann1_ann2_modelC") == 0){
        
        // Model C: 4-x-3_6-y-1
        // The 1st ANN (4-x-3), used for the learning phase (coordinates system, flag 2D), has:
        //   - inputs: a signal (chemicals_to_spread) from each neighbor. ann1 input = neighbors_states = [ signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ]
        //   - output: a signal to spread to neighbors, and two phenotype values x and y. ann1 output = [ signal_xy, x, y ]
        // The 2nd ANN (6-y-1), used to learn the target flag (two-bands, centered-half-discs, ...), has:
        //   - inputs: x and y from the coordinate system, plus the ann1 neighbors_states. ann2 input = [ x, y, signal_xy_N, signal_xy_W, signal_xy_E, signal_xy_S ]
        //   - output: one phenotype. ann2 output = [ p ]
        // The final state for an agent, is [signal_xy from ann1, p from ann2]
        // NB: ann1 phenotypes (x, y) are rescaled from (-1,1) to (0,1); ann2 phenotypes (p) will be rescaled in agent2Outputs
        
        if (!( (strcmp(ANN1_AGENT, "agentCoordinates_gradient") == 0) && ( (strcmp(ANN2_AGENT, "agent2Outputs") == 0) || (strcmp(ANN2_AGENT, "agent2Outputs_RGB") == 0) ) )){
            printf("Error in compute_robot_state ann1_ann2_modelPIX-C: ANN2_AGENT not correct, check controller parameters.");
            exit(1);
        }

        // Here we prepare the ann2_input_layer
        ann2_input_layer[0] = (ann1_output_layer[1] + 1.0) / 2.0; // rescale phenotype x in (-1,1) to (0,1)
        ann2_input_layer[1] = (ann1_output_layer[2] + 1.0) / 2.0; // rescale phenotype y in (-1,1) to (0,1)
        for (int i = 0; i < ann1_input_size; i++) {
            ann2_input_layer[2 + i] = ann1_input_layer[i];
        }

        fnn_predict(
            ann2_input_layer,
            ann2_output_layer,
            ann2_nb_layers,
            ann2_weights_sizes,
            ann2_weights,
            ann2_biases,
            ANN2_ACTIVATION_FUNCTION_ARRAY,
            ann2_input_size,
            ann2_output_size
        );

        for (i=0; i<CHEMICALS_TO_SPREAD_SIZE; i++){
            my_chemicals_to_spread[i] = ann1_output_layer[i]; // ANN1 returns the chemicals to share with my neighbors via msg, for all setups
        }
        for (i=0; i<PHENOTYPE_SIZE; i++){
            my_phenotype[i] = (ann2_output_layer[ann2_output_size - PHENOTYPE_SIZE + i] + 1.0) / 2.0; // rescale phenotype x in (-1,1) to (0,1);
        }
    }
}

//-------------------------------------------------------------

void print_all_chemicals_infos(uint16_t my_pogobot_id, uint16_t age){
    // printf("\n[I'm Pogobot %d] Current state:\n", my_pogobot_id);
    // printf("\t[AGE] Age = %d\n", age);

    // printf("\t[INPUT FNN] ann1_input_layer = [ ");
    // for (int i = 0; i < INPUT_SIZE; i++) {
    //     print_double(ann1_input_layer[i]); printf(" ");
    // }
    // printf("]\n");

    // printf("\t[OUTPUT FNN] ann_output_layer = [ ");
    // for (int i = 0; i < OUTPUT_SIZE; i++) {
    //     print_double(ann_output_layer[i]); printf(" ");
    // }
    // printf("]\n");

    // printf("\t[MY EXT.CHEMICALS] my_chemicals_to_spread = [ ");
    // for (int i = 0; i < CHEMICALS_TO_SPREAD_SIZE; i++) {
    //     print_double(my_chemicals_to_spread[i]); printf(" ");
    // }
    // printf("]\n");

    // printf("\t[PHENOTYPE] Phenotype value = "); print_double(my_phenotype); printf("\n");
}

//-------------------------------------------------------------

void print_double(double val) {
    
    // Handle negative values
    if (val < 0) {
        putchar('-');
        val = -val;
    }

    // Get integer and decimal parts
    int int_part = (int)val;
    double dec_part = val - int_part;

    // Adjust the decimal part to the required precision
    int scaled_dec_part = (int)(dec_part * pow(10, NB_DIGIT) + 0.5);  // Rounding

    // Print the result
    printf("%d.%0*d", int_part, NB_DIGIT, scaled_dec_part);
}

//-------------------------------------------------------------

// power_of_ten can only be 1, 10, 100, 1000... etc
int log_10(int power_of_ten){
    int pw = 0;
    int i = 1;
    while (i < power_of_ten){
        pw++;
        i*=10;
    }

    return pw;
}

//-------------------------------------------------------------

int double_get_decimal_part(double value){
	return (int)((value-(int)value)*PRECISION);
}

//-------------------------------------------------------------

void double_get_decimal_part_string(double value, char *res){
    
    int dec_part = double_get_decimal_part(value);
    
    int j=10;
    res[NB_DIGIT] = '\0';
    for (int i=NB_DIGIT-1; i>=0 && j<=PRECISION; i--){
        res[i] = (char)((int)((dec_part%j)/(j/10))) + '0';
        j*=10;
    }
}

//-------------------------------------------------------------

int double_get_integer_part(double value) {
	return (int)value;
}

// ===================== Fonctions Utiles =====================
static uint32_t rand_between(uint32_t min, uint32_t max) {
    return min + (rand() % (max - min + 1));
}
// ===================== Moteurs =====================
// Les fonctions ci-dessous masquent le détail bas niveau (sens + puissance) pour exprimer des intentions : avancer, reculer, tourner, etc.

static inline void motors_stop(void){
    pogobot_motor_power_set(motorL, 0);                             // Met puissance à 0 → arrêt moteur gauche.
    pogobot_motor_power_set(motorR, 0);                             // Idem à droite.
}
static void motors_forward(void) {
    pogobot_motor_dir_set(motorL, dirL);
    pogobot_motor_dir_set(motorR, dirR);
    pogobot_motor_power_set(motorL, pwmLr);
    pogobot_motor_power_set(motorR, pwmRr);
    pogobot_motor_power_set(motorB, 0);
}

static void motors_backward(void) {
    pogobot_motor_dir_set(motorL, 1 - dirL);
    pogobot_motor_dir_set(motorR, 1 - dirR);
    pogobot_motor_power_set(motorL, pwmLr);
    pogobot_motor_power_set(motorR, pwmRr);
    pogobot_motor_power_set(motorB, 0);
}

static void motors_turn_left(void) {
    pogobot_motor_dir_set(motorL, 1 - dirL);
    pogobot_motor_dir_set(motorR, dirR);
    pogobot_motor_power_set(motorL, pwmLt);
    pogobot_motor_power_set(motorR, pwmRt);
    pogobot_motor_power_set(motorB, 0);
}

static void motors_turn_right(void) {
    pogobot_motor_dir_set(motorL, dirL);
    pogobot_motor_dir_set(motorR, 1 - dirR);
    pogobot_motor_power_set(motorL, pwmLt);
    pogobot_motor_power_set(motorR, pwmRt);
    pogobot_motor_power_set(motorB, 0);
}

static void motors_turn_slow_right(void) {
    pogobot_motor_dir_set(motorL, dirL);
    pogobot_motor_dir_set(motorR, 1 - dirR);
    pogobot_motor_power_set(motorL, PWR_SLOW);
    pogobot_motor_power_set(motorR, PWR_SLOW);
    pogobot_motor_power_set(motorB, 0);
}

static void motors_slow_forward(void) {
    pogobot_motor_dir_set(motorL, dirL);
    pogobot_motor_dir_set(motorR, dirR);
    pogobot_motor_power_set(motorL, PWR_SLOW);
    pogobot_motor_power_set(motorR, PWR_SLOW);
    pogobot_motor_power_set(motorB, 0);
}

static void app_init(void){
    pogobot_infrared_set_power(INFRARED_POWER);                       // Fixe la puissance IR selon notre réglage.
    if(robot_identifier==0){ 
        mode = MODE_ROOT;
        pos.row = 0;
        pos.col = 0;
    }
    else{
        uint8_t mem[3] = {0};                                             // Petit tampon (3 octets) pour récup calibration moteurs.
        pogobot_motor_dir_mem_get(mem);                                   // Lecture calibration (SDK). Convention : mem[0]=droite, mem[1]=gauche.
        dirR = mem[0];                                                  // Applique la direction logique du moteur droit.
        dirL = mem[1];                                                  // Applique la direction logique du moteur gauche.

        
        uint16_t pwr[3] = {0};
        if (pogobot_motor_power_mem_get(pwr) == 0) {
            //pwmRt = pwr[0] * FACT1;
            //pwmLt = pwr[1] * FACT1;
            //pwmRr = pwr[0] * FACT;
            //pwmLr = pwr[1] * FACT;
            pwmRt = MOTOR_POWER * FACT1; 
            pwmLt = MOTOR_POWER * FACT1;
            pwmRr = MOTOR_POWER * FACT;
            pwmLr = MOTOR_POWER * FACT;


            printf("Puissances R : L=%lu, R=%lu\n", (unsigned long)pwmLr, (unsigned long)pwmRr);
            printf("Puissances T : L=%lu, R=%lu\n", (unsigned long)pwmLt, (unsigned long)pwmRt);
        } else {
            printf("Erreur de lecture de la mémoire moteur ! On utilise des valeurs par défault\n");
            pwmRt = MOTOR_POWER * FACT1; 
            pwmLt = MOTOR_POWER * FACT1;
            pwmRr = MOTOR_POWER * FACT;
            pwmLr = MOTOR_POWER * FACT;
            
        }

        mode = DEFAULT_MODE;                                             // On démarre en exploration.

        // Run & tumble : on tire les premières durées et directions.
        rt_phase = RT_PHASE_RUN;                                        // Première phase = RUN (avancer/reculer).
        pogobot_stopwatch_reset(&rt_phase_timer);                               // Timer de phase RT démarré maintenant.
        
        rt_duration_ms = rand_between(RT_RUN_MIN_MS, RT_RUN_MAX_MS);
        a_ttl = A_TTL_MS;
        rt_tumble_left   = rand() % 2;                           // TUMBLE initial : gauche si bit aléatoire=1 (sinon droite).
        rt_run_backward  = rand() % 2;                           // RUN initial : arrière si bit=1 (sinon avant).
        check_phase_ttl = CHECK_PHASE_MS; // Durant pendant l'étape de confirmation de connexion ou il restera dans sa phase
        con_ttl = CON_TIMER_MS; // Durant de forward pendant la tentative de connexion
        check_ttl = CHECK_TIMER_MS;
        seek_ttl = SEEK_MS;
        max_ttl = MAX_MS;
        finish_ttl = FINISH_MS;

        check_phase = CHECK_PHASE_FW;
        rec_input_ttl = REC_INPUT_TIMER_MS;
        // Feedback & moteurs : LED "run", et on démarre en avant/arrière selon le tirage.
        if (rt_run_backward) motors_backward(); else motors_forward();
    }
}

static void update_run_tumble(void) {
    if(MESSAGE==1){
        pogobot_infrared_update();
        for(int i=0;i<5;i++) pogobot_led_setColors(0,0,0,i);
        if (pogobot_infrared_message_available()) {
            //pogobot_led_setColor(purple.r, purple.g, purple.b);
            int msg_rcv = 0;
            while (pogobot_infrared_message_available() && msg_rcv < MAX_NB_OF_MSG){
                message_t mr;
                pogobot_infrared_recover_next_message(&mr);
                uint8_t ir = mr.header._receiver_ir_index + 1;

                //affichage reception leds
                uint8_t my_led_index2 = mr.header._receiver_ir_index + 1;
                pogobot_led_setColors(0, 0, 25, ir);
                if (my_led_index2 > 0) // default value is -1. my_led_index2 value is 1, 2, 3 or 4 in case of IR reception activity
                {
                    for (int i  = 1; i < 5 ; i++)
                
                    {
                        pogobot_led_setColors(0, 0, 0, i);
                    }
                    my_led_index2 = -1;
                }
            
                grid_msg_t recv;
                memcpy(&recv, mr.payload, GRID_MSG_SIZE);
                if(mr.header._sender_ir_index == 2 && recv.col == 0){
                    if(mr.header._receiver_ir_index == 0){
                        mode = MODE_CONNECTION_COL;
                        pogobot_stopwatch_reset(&con_timer); 
                        pogobot_stopwatch_reset(&rec_input_timer);
                        break;
                    }
                    pogobot_led_setColors(purple.r, purple.g, purple.b, ir);
                    msg_rcv++;
                    mode = MODE_ALIGN;
                    pogobot_stopwatch_reset(&a_timer);
                    motors_stop();
                    break;
                }
                else if(mr.header._sender_ir_index == 1  && recv.row == 0){
                    if(mr.header._receiver_ir_index == 3){
                        mode = MODE_CONNECTION_ROW;
                        pogobot_stopwatch_reset(&con_timer);
                        pogobot_stopwatch_reset(&rec_input_timer);
                        break;
                    }
                    pogobot_led_setColors(purple.r, purple.g, purple.b, ir);
                    msg_rcv++;
                    mode = MODE_ALIGN;
                    pogobot_stopwatch_reset(&a_timer);
                    motors_stop();
                    break;
                }
                else{
                    if(mr.header._sender_ir_index == 2) {
                        pogobot_stopwatch_reset(&a_timer);
                        mode = MODE_ALIGN_TO_SEEK;
                        break;
                    }
                    else {
                    if(mr.header._receiver_ir_index == 3 && mr.header._sender_ir_index == 1){
                        mode = MODE_CONNECTION_ROW;
                        pogobot_stopwatch_reset(&con_timer); 
                        pogobot_stopwatch_reset(&rec_input_timer);
                        break;
                    }
                    pogobot_led_setColors(purple.r, purple.g, purple.b, ir);
                    msg_rcv++;
                    mode = MODE_ALIGN;
                    pogobot_stopwatch_reset(&a_timer);
                    motors_stop();
                    break;
                    }
                }
            }
            pogobot_infrared_clear_message_queue();
        }
    }
    // Calcul du temps écoulé en ms
    uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&rt_phase_timer) / 1000);

    if (rt_phase == RT_PHASE_RUN) {
        pogobot_led_setColor(cyan.r, cyan.g, cyan.b);
        if (rt_run_backward) motors_backward();                      // Si le tirage aléatoire a choisi "reculer"...
        else                   motors_forward();

        if (elapsed >= rt_duration_ms) {
            rt_phase = RT_PHASE_TUMBLE;
            pogobot_stopwatch_reset(&rt_phase_timer);
            rt_duration_ms = rand_between(RT_TUMBLE_MIN_MS, RT_TUMBLE_MAX_MS);
            rt_tumble_left = rand() % 2;;
        }
    } else {
        //pogobot_led_setColor(yellow.r, yellow.g, yellow.b);
        if (rt_tumble_left > 0.33) motors_turn_left(); else motors_turn_right();

        if (elapsed >= rt_duration_ms) {
            rt_phase = RT_PHASE_RUN;
            pogobot_stopwatch_reset(&rt_phase_timer);
            rt_duration_ms = rand_between(RT_RUN_MIN_MS, RT_RUN_MAX_MS);
            rt_run_backward = rand() % 2; 
        }
    }
}

static void update_root(void){
    pogobot_led_setColor(green.r, green.g, green.b);
    if(MESSAGE == 1){
            pogobot_infrared_update();
            if (pogobot_infrared_message_available()) {
                int msg_rcv = 0;
                while (pogobot_infrared_message_available() && msg_rcv < MAX_NB_OF_MSG){
                    message_t mr;
                    pogobot_infrared_recover_next_message(&mr);
                    if(mr.header._sender_ir_index == 0 || mr.header._sender_ir_index == 3){
                        mode = MODE_FLAG;
                        pogobot_stopwatch_reset(&finish_timer);
                        motors_stop();
                        break;
                    }
                } 
            }
        }
    if(rand()*100<=P_MSG){
        pogobot_led_setColors(red.r, red.g, red.b, 3);
        pogobot_infrared_sendLongMessage_uniSpe(2, (uint8_t*)&pos, GRID_MSG_SIZE);
        pogobot_infrared_sendLongMessage_uniSpe(1, (uint8_t*)&pos, GRID_MSG_SIZE);
        pogobot_led_setColors(red.r, red.g, red.b, 2);
    }
}

static void update_align(void){
    if(MESSAGE==1){
        pogobot_infrared_update();
        if (pogobot_infrared_message_available()) {
            for(int i=0;i<5;i++) pogobot_led_setColors(0,0,0,i);
            int msg_rcv = 0;
            while (pogobot_infrared_message_available() && msg_rcv < MAX_NB_OF_MSG){
                message_t mr;
                pogobot_infrared_recover_next_message(&mr);
                uint8_t ir = mr.header._receiver_ir_index + 1;

                // affichage reception leds
                pogobot_led_setColors(0, 0, 25, ir);
                if (ir > 0)
                {
                    for (int i = 1; i < 5; i++)
                    {
                        pogobot_led_setColors(0, 0, 0, i);
                    }
                    ir = -1;
                }

                grid_msg_t recv;
                memcpy(&recv, mr.payload, GRID_MSG_SIZE);

                if(mr.header._sender_ir_index == 2 && recv.col == 0){
                    if(mr.header._receiver_ir_index == 0){
                        mode = MODE_CONNECTION_COL;
                        pogobot_stopwatch_reset(&con_timer); 
                        pogobot_stopwatch_reset(&rec_input_timer);
                        break;
                    }
                    pogobot_led_setColors(purple.r, purple.g, purple.b, ir);
                    msg_rcv++;
                    mode = MODE_ALIGN;
                    pogobot_stopwatch_reset(&a_timer);
                    motors_stop();
                    break;
                }
                else if(mr.header._sender_ir_index == 1 && recv.row == 0){
                    if(mr.header._receiver_ir_index == 3){
                        mode = MODE_CONNECTION_ROW;
                        pogobot_stopwatch_reset(&con_timer); 
                        pogobot_stopwatch_reset(&rec_input_timer);
                        break;
                    }
                    pogobot_led_setColors(purple.r, purple.g, purple.b, ir);
                    msg_rcv++;
                    mode = MODE_ALIGN;
                    pogobot_stopwatch_reset(&a_timer);
                    motors_stop();
                    break;
                }
                else{
                    if(mr.header._receiver_ir_index == 3 && mr.header._sender_ir_index == 1){
                        mode = MODE_CONNECTION_ROW;
                        pogobot_stopwatch_reset(&con_timer); 
                        pogobot_stopwatch_reset(&rec_input_timer);
                        break;
                    }
                    pogobot_led_setColors(purple.r, purple.g, purple.b, ir);
                    msg_rcv++;
                    mode = MODE_ALIGN;
                    pogobot_stopwatch_reset(&a_timer);
                    motors_stop();
                    break;
                }
            }
            pogobot_infrared_clear_message_queue();
        }
    }

    for(int i=0;i<5;i++) pogobot_led_setColors(0,0,0,i);
    pogobot_led_setColor(purple.r, purple.g, purple.b);

    uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&a_timer) / 1000);
    if(elapsed >= a_ttl){
        mode = MODE_NORMAL;
        return;
    }

    motors_turn_slow_right();
}

static void update_connection_col(void){
    if(MESSAGE == 1){
        pogobot_infrared_update();
        
        if (pogobot_infrared_message_available()) {
            int msg_rcv = 0;
            while (pogobot_infrared_message_available() && msg_rcv < MAX_NB_OF_MSG){
                
                message_t mr;
                pogobot_infrared_recover_next_message(&mr);
                if(mr.header._receiver_ir_index  == 0){
                    pogobot_stopwatch_reset(&rec_input_timer);
                    grid_msg_t recv;
                    memcpy(&recv, mr.payload, GRID_MSG_SIZE);
                    pos.row = recv.row+1;
                    pos.col = recv.col;
                }
            } 
            pogobot_infrared_clear_message_queue();
        }
        uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&rec_input_timer) / 1000);
        if (elapsed > rec_input_ttl) mode = MODE_NORMAL;
    }
    pogobot_led_setColor(25,25,25);
    // Calcul du temps écoulé en ms
    uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&con_timer) / 1000);
    if (elapsed < con_ttl) motors_forward();
    else{
        mode=MODE_CHECK_CONNECTION_COL;
        pogobot_stopwatch_reset(&check_phase_timer);
        pogobot_stopwatch_reset(&check_timer);
    }
}

static void update_connection_row(void){
    if(MESSAGE == 1){
        pogobot_infrared_update();
        
        if (pogobot_infrared_message_available()) {
            int msg_rcv = 0;
            while (pogobot_infrared_message_available() && msg_rcv < MAX_NB_OF_MSG){
                
                message_t mr;
                pogobot_infrared_recover_next_message(&mr);
                if(mr.header._receiver_ir_index  == 3){
                    pogobot_stopwatch_reset(&rec_input_timer);
                    grid_msg_t recv;
                    memcpy(&recv, mr.payload, GRID_MSG_SIZE);
                    pos.row = recv.row;
                    pos.col = recv.col+1;
                }
            } 
            pogobot_infrared_clear_message_queue();
        }
        uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&rec_input_timer) / 1000);
        if (elapsed > rec_input_ttl) mode = MODE_NORMAL;
    }
    pogobot_led_setColor(25,25,25);
    // Calcul du temps écoulé en ms
    uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&con_timer) / 1000);
    if (elapsed < con_ttl) motors_forward();
    else{
        mode=MODE_CHECK_CONNECTION_ROW;
        pogobot_stopwatch_reset(&check_phase_timer);
        pogobot_stopwatch_reset(&check_timer);
    }

}

static void update_check_connection_col(void){
    pogobot_led_setColor(yellow.r,yellow.g,yellow.b);
    if(MESSAGE == 1){
        pogobot_infrared_update();
        
        if (pogobot_infrared_message_available()) {
            int msg_rcv = 0;
            while (pogobot_infrared_message_available() && msg_rcv < MAX_NB_OF_MSG){
                message_t mr;
                pogobot_infrared_recover_next_message(&mr);
                if(mr.header._receiver_ir_index  == 0){
                    pogobot_stopwatch_reset(&rec_input_timer);
                    pogobot_infrared_clear_message_queue();
                }
            } 
        }
        uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&rec_input_timer) / 1000);
        if (elapsed > rec_input_ttl) mode = MODE_NORMAL;
    }
   uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&check_timer) / 1000);
   if (elapsed < check_ttl){
        if(check_phase == CHECK_PHASE_FW){
            uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&check_phase_timer) / 1000);
            if (elapsed < check_phase_ttl) motors_slow_forward();
            else{
                check_phase = CHECK_PHASE_TURN;
                pogobot_stopwatch_reset(&check_phase_timer);
            }
    }
        else{
            uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&check_phase_timer) / 1000);
            if (elapsed < check_phase_ttl) motors_turn_slow_right();
            else{
                check_phase = CHECK_PHASE_FW;
                pogobot_stopwatch_reset(&check_phase_timer);
            }
        }
   }
   else{
        mode = MODE_CONNECTED;
   }
}

static void update_check_connection_row(void){
    pogobot_led_setColor(yellow.r,yellow.g,yellow.b);
    if(MESSAGE == 1){
        pogobot_infrared_update();
        
        if (pogobot_infrared_message_available()) {
            int msg_rcv = 0;
            while (pogobot_infrared_message_available() && msg_rcv < MAX_NB_OF_MSG){
                message_t mr;
                pogobot_infrared_recover_next_message(&mr);
                if(mr.header._receiver_ir_index  == 3){
                    pogobot_stopwatch_reset(&rec_input_timer);
                    pogobot_infrared_clear_message_queue();
                }
            } 
        }
        uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&rec_input_timer) / 1000);
        if (elapsed > rec_input_ttl) mode = MODE_NORMAL;
    }
   uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&check_timer) / 1000);
   if (elapsed < check_ttl){
        if(check_phase == CHECK_PHASE_FW){
            uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&check_phase_timer) / 1000);
            if (elapsed < check_phase_ttl) motors_slow_forward();
            else{
                check_phase = CHECK_PHASE_TURN;
                pogobot_stopwatch_reset(&check_phase_timer);
            }
    }
        else{
            uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&check_phase_timer) / 1000);
            if (elapsed < check_phase_ttl) motors_turn_slow_right();
            else{
                check_phase = CHECK_PHASE_FW;
                pogobot_stopwatch_reset(&check_phase_timer);
            }
        }
   }
   else{
        mode = MODE_CONNECTED;
   }
}

static void update_connected(void){
    pogobot_led_setColor(green.r, green.g, green.b);
    motors_stop();
    if(MESSAGE == 1){
            pogobot_infrared_update();
            
            if (pogobot_infrared_message_available()) {
                int msg_rcv = 0;
                while (pogobot_infrared_message_available() && msg_rcv < MAX_NB_OF_MSG){
                    message_t mr;
                    pogobot_infrared_recover_next_message(&mr);
                    if(mr.header._sender_ir_index == 0 || mr.header._sender_ir_index == 3){
                        mode = MODE_FLAG;
                        pogobot_stopwatch_reset(&finish_timer);
                        motors_stop();
                        break;
                    }
                    if(pos.col == 0){
                        if(mr.header._receiver_ir_index  == 0){
                        pogobot_stopwatch_reset(&rec_input_timer);
                        pogobot_infrared_clear_message_queue();
                        }
                    }
                    if(pos.row == 0){
                        if(mr.header._receiver_ir_index  == 3){
                        pogobot_stopwatch_reset(&rec_input_timer);
                        pogobot_infrared_clear_message_queue();
                        }
                    }
                    else{
                        if(mr.header._receiver_ir_index  == 3){
                        pogobot_stopwatch_reset(&rec_input_timer);
                        pogobot_infrared_clear_message_queue();
                        }
                    }
                } 
            }
            uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&rec_input_timer) / 1000);
            if (elapsed > rec_input_ttl) mode = MODE_NORMAL;
        }

    if(check_phase == CHECK_PHASE_FW){
            uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&check_phase_timer) / 1000);
            if (elapsed < check_phase_ttl) motors_slow_forward();
            else{
                check_phase = CHECK_PHASE_TURN;
                pogobot_stopwatch_reset(&check_phase_timer);
            }
    } else{
        uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&check_phase_timer) / 1000);
            if (elapsed < check_phase_ttl) motors_stop();
            else{
                check_phase = CHECK_PHASE_FW;
                pogobot_stopwatch_reset(&check_phase_timer);
            }
    }
    for(int i=0;i<5;i++) pogobot_led_setColors(0,0,0,i);

    if(rand()*100<=P_MSG ){
        if(pos.row < MAX_GRID) {
            pogobot_infrared_sendLongMessage_uniSpe(2, (uint8_t*)&pos, GRID_MSG_SIZE);
            pogobot_led_setColors(red.r, red.g, red.b, 3);
        }
        if(pos.col < MAX_GRID) {;
            pogobot_infrared_sendLongMessage_uniSpe(1, (uint8_t*)&pos, GRID_MSG_SIZE);
            pogobot_led_setColors(red.r, red.g, red.b, 2);
        }
        printf("%u %u", pos.row, pos.col);

    }
    if(pos.row == MAX_GRID && pos.col == MAX_GRID){
        pogobot_stopwatch_reset(&finish_timer);
        mode = MODE_FLAG;
    }
}

static void update_align_to_seek(void){
    if(MESSAGE==1){
        pogobot_infrared_update();
        if (pogobot_infrared_message_available()) {
            for(int i=0;i<5;i++) pogobot_led_setColors(0,0,0,i);
            //pogobot_led_setColor(purple.r, purple.g, purple.b);
            int msg_rcv = 0;
            while (pogobot_infrared_message_available() && msg_rcv < MAX_NB_OF_MSG){
                message_t mr;
                pogobot_infrared_recover_next_message(&mr);
                uint8_t ir = mr.header._receiver_ir_index + 1;

                //affichage reception leds
                uint8_t my_led_index2 = mr.header._receiver_ir_index + 1;
                pogobot_led_setColors(0, 0, 25, ir);
                if (my_led_index2 > 0) // default value is -1. my_led_index2 value is 1, 2, 3 or 4 in case of IR reception activity
                {
                    for (int i  = 1; i < 5 ; i++)
                    {
                        pogobot_led_setColors(0, 0, 0, i);
                    }
                    my_led_index2 = -1;
                }

                grid_msg_t recv;
                memcpy(&recv, mr.payload, GRID_MSG_SIZE);
                if(mr.header._receiver_ir_index == 1){
                    mode = MODE_SEEK_ROOT;
                    pogobot_stopwatch_reset(&seek_timer);
                    pogobot_stopwatch_reset(&max_timer);
                    pogobot_infrared_clear_message_queue();
                    break;
                }
                }
            }
        }
    pogobot_led_setColor(purple.r, purple.g, purple.b);
    uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&a_timer) / 1000);
    if(elapsed >= a_ttl){
        mode = MODE_NORMAL;
        return;
    }
    motors_turn_slow_right();

}

static void update_seek_root(void){
    pogobot_led_setColor(light_pink.r,light_pink.g,light_pink.b);
    if(MESSAGE == 1){
        pogobot_infrared_update();
        
        if (pogobot_infrared_message_available()) {
            int msg_rcv = 0;
            while (pogobot_infrared_message_available() && msg_rcv < MAX_NB_OF_MSG){
                message_t mr;
                pogobot_infrared_recover_next_message(&mr);
                grid_msg_t recv;
                memcpy(&recv, mr.payload, GRID_MSG_SIZE);
                if(recv.col == 0 && mr.header._sender_ir_index == 2){
                    pogobot_stopwatch_reset(&a_timer);
                    mode = MODE_ALIGN;
                    break;
                }
                else if(mr.header._sender_ir_index == 1){
                    mode = MODE_ALIGN;
                    break;
                }
                else if(mr.header._sender_ir_index  == 2){
                    pogobot_stopwatch_reset(&seek_timer);
                    break;
                }
                pogobot_infrared_clear_message_queue();
            } 
        }
        uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&seek_timer) / 1000);
        uint32_t elapsed_max = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&max_timer) / 1000);
        if (elapsed > seek_ttl || elapsed_max > max_ttl) mode = MODE_NORMAL;
        else motors_slow_forward();
    }
}

static void update_flag(void){
    pogobot_led_setColor(yellow.r, yellow.g, yellow.b);
    for(int i=0;i<5;i++) pogobot_led_setColors(0,0,0,i);
    pogobot_infrared_sendLongMessage_uniSpe(0, (__uint8_t*)4, (__uint16_t) 4);
    pogobot_led_setColors(red.r, red.g, red.b, 1);
    pogobot_infrared_sendLongMessage_uniSpe(3, (__uint8_t*)4, (__uint16_t) 4);
    pogobot_led_setColors(red.r, red.g, red.b, 4);
    uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&finish_timer) / 1000);
    if(elapsed > finish_ttl){ 
        mode = MODE_FLAG_LIGHT;
        flag_light_initialized = false;
}
}


static void update_flag_light(void){

    if(!flag_light_initialized){
        timeout_NWES[0] = timeout_N;
        timeout_NWES[1] = timeout_W;
        timeout_NWES[2] = timeout_E;
        timeout_NWES[3] = timeout_S;

        my_pogobot_id = pogobot_helper_getid();
        age = 0;
        counter_rcvd_msgs = 0;
        my_led_index = -1;
        color_index = -1;

        for (int i=0; i<ann1_input_size; i++)
            ann1_input_layer[i] = 0.0;

        for (int i=0; i<CHEMICALS_TO_SPREAD_SIZE; i++)
            my_chemicals_to_spread[i] = 0.0;
    
        for (int i=0; i<PHENOTYPE_SIZE; i++)
            my_phenotype[i] = 0.0;

        for (uint16_t i=0; i != MSG_SIZE; i++)
            msg_from_neighbor.msg_array[i] = 0;

        pogobot_stopwatch_reset(&timeout_phenotypic_freshness);
        pogobot_stopwatch_reset(&timeout_N);
        pogobot_stopwatch_reset(&timeout_W);
        pogobot_stopwatch_reset(&timeout_E);
        pogobot_stopwatch_reset(&timeout_S);

        flag_light_initialized = true;
    }

    motors_stop();

    if (((double)rand() / (double)RAND_MAX) < p_change_led_color){
        age++;
        update_robot_state(age);
        pogobot_stopwatch_reset(&timeout_phenotypic_freshness);

        if (DEBUG_LEVEL == 1 || DEBUG_LEVEL == 99) printf("\n[I'm Pogobot %d] [AGE] Incrementing age to %d because I'm lucky\n", my_pogobot_id, age);
        if (DEBUG_LEVEL == 3 || DEBUG_LEVEL == 99) {printf("\n[I'm Pogobot %d] [INPUT FNN] Chemicals and phenotype updated\n", my_pogobot_id); print_all_chemicals_infos(my_pogobot_id, age);}
    }

    for (nb_ir=0; nb_ir<4; nb_ir++){
        seconds = pogobot_stopwatch_get_elapsed_microseconds(&timeout_NWES[nb_ir]) / 1000000;
        if (seconds >= max_chemicals_to_spread_validity_time_seconds){
            for (int i=nb_ir*CHEMICALS_TO_SPREAD_SIZE; i<(nb_ir*CHEMICALS_TO_SPREAD_SIZE)+CHEMICALS_TO_SPREAD_SIZE; i++)
                ann1_input_layer[i] = 0.0;
                
            pogobot_stopwatch_reset(&timeout_NWES[nb_ir]);

            if (DEBUG_LEVEL == 3 || DEBUG_LEVEL == 99) { printf("\n[I'm Pogobot %d] [INPUT FNN] Reset of neighbors external chemicals on IR n.%d.\n", my_pogobot_id, nb_ir); for (int i = 0; i < ann1_input_size; i++) {printf("\tann1_input_layer[%d] = ", i); print_double(ann1_input_layer[i]); printf("\n");}}
        }
    }
        
    seconds = pogobot_stopwatch_get_elapsed_microseconds(&timeout_phenotypic_freshness) / 1000000;
    if (seconds >= phenotypic_freshness_time_seconds){
        pogobot_led_setColor(0, 0, 0);
    } else if (seconds >= hanabi_blinking_time_seconds){ 
        display_phenotype(my_phenotype);
    }

    pogobot_infrared_update();

    if (pogobot_infrared_message_available()){
        counter_rcvd_msgs = 0;
            
        while (pogobot_infrared_message_available() && counter_rcvd_msgs < MAX_NB_OF_MSG) {                
            message_t mr;
            pogobot_infrared_recover_next_message(&mr);

            if (mr.header._packet_type != ir_t_user) {
                printf("[I'm Pogobot %d] [RECV] This message is discarded because it didn't come from a Pogobot\n", my_pogobot_id);
                continue;
            }
       
            for (uint16_t i = 0; i != MSG_SIZE; i++)
                msg_from_neighbor.msg_array[i] = mr.payload[i];

            printf("\treceived msg = [ "); for (int i = 0; i <CHEMICALS_TO_SPREAD_SIZE; i++) {print_double(msg_from_neighbor.msg_values.chemicals_to_spread[i]); printf(" ");} printf("]\n");

            my_receiver_ir_index = mr.header._receiver_ir_index;
            my_led_index = my_receiver_ir_index + 1;
            pogobot_led_setColors(0, 0, 25, my_led_index);

            if (DEBUG_LEVEL == 2 || DEBUG_LEVEL == 99) {printf("\n[I'm Pogobot %d] [MSG] New message received on IR n.%d. Received msg = [ \n", my_pogobot_id, my_receiver_ir_index); for (int i = 0; i <CHEMICALS_TO_SPREAD_SIZE; i++) {print_double(msg_from_neighbor.msg_values.chemicals_to_spread[i]); printf(" ");} printf("]\n");}

            switch (my_receiver_ir_index){
                case 0:
                    starting_index_NWES = 0;
                    break;
                case 1:
                    starting_index_NWES = 2*CHEMICALS_TO_SPREAD_SIZE;
                    break;
                case 2:
                    starting_index_NWES = 3*CHEMICALS_TO_SPREAD_SIZE;
                    break;
                case 3:
                    starting_index_NWES = CHEMICALS_TO_SPREAD_SIZE;
                    break;
            }

            for(int j = 0; j < CHEMICALS_TO_SPREAD_SIZE; j++){
                ann1_input_layer[starting_index_NWES + j] = msg_from_neighbor.msg_values.chemicals_to_spread[j];
            }
            pogobot_stopwatch_reset(&timeout_NWES[my_receiver_ir_index]);

            if (DEBUG_LEVEL == 3 || DEBUG_LEVEL == 99) {printf("\n[I'm Pogobot %d] [MSG] New neighbors external chemicals received on IR n.%d\n", my_pogobot_id, my_receiver_ir_index); printf("\treceived msg = [ "); for (int i = 0; i <CHEMICALS_TO_SPREAD_SIZE; i++) {print_double(msg_from_neighbor.msg_values.chemicals_to_spread[i]); printf(" ");} printf("]\n"); printf("\tann1_input_layer = [ "); for (int i = 0; i < ann1_input_size; i++) {print_double(ann1_input_layer[i]); printf(" ");} printf("]\n");}

            if (msg_from_neighbor.msg_values.age > age && color_index < nb_rgb_colors && (msg_from_neighbor.msg_values.age - age) < 10000) {
                age = msg_from_neighbor.msg_values.age;
                update_robot_state(age);
                pogobot_stopwatch_reset(&timeout_phenotypic_freshness);

                if (DEBUG_LEVEL == 3 || DEBUG_LEVEL == 99) {printf("\n[I'm Pogobot %d] [INPUT FNN] Chemicals and phenotype updated\n", my_pogobot_id); print_all_chemicals_infos(my_pogobot_id, age);}
            }
            counter_rcvd_msgs++;
        }
    }
    else {
        if (my_led_index > 0){
            for (int i  = 1; i < 5 ; i++){
                pogobot_led_setColors(0, 0, 0, i);
            }
            my_led_index = -1;
        }
    }
    pogobot_infrared_clear_message_queue();

    if (((double)rand() / (double)RAND_MAX) < p_send_per_step){

        msg_from_neighbor.msg_values.sender_id = my_pogobot_id;
        msg_from_neighbor.msg_values.age = age;
            
        for (uint16_t i = 0; i != CHEMICALS_TO_SPREAD_SIZE; i++){
            msg_from_neighbor.msg_values.chemicals_to_spread[i] = my_chemicals_to_spread[i];
        }

        for (uint16_t i = 0; i != MSG_SIZE; i++)
            data[i] = msg_from_neighbor.msg_array[i];

        pogobot_infrared_sendLongMessage_omniGen((uint8_t *)(data), MSG_SIZE);
        if (((double)rand() / (double)RAND_MAX) < 0.001){
            if (DEBUG_LEVEL == 3 || DEBUG_LEVEL == 99) {printf("\n[I'm Pogobot %d] [MSG] New message sent in all directions.\n", my_pogobot_id); for (uint16_t i = 0; i != CHEMICALS_TO_SPREAD_SIZE; i++) {printf("\tmsg_from_neighbor.msg_values.chemicals_to_spread[%d] = ", i); print_double(msg_from_neighbor.msg_values.chemicals_to_spread[i]); printf("\n");}}
        }
    }
}

static void update_mode(void){
    //if(MESSAGE==1){
    //    update_message();
    //}
    switch(mode){
        case MODE_NORMAL:
            update_run_tumble();
            break;
        case MODE_ROOT:
            update_root();
            break;
        case MODE_ALIGN:
            update_align();
            break;
        case MODE_CONNECTION_COL:
            update_connection_col();
            break;
        
        case MODE_CONNECTION_ROW:
            update_connection_row();
            break;
        case MODE_CHECK_CONNECTION_ROW:
            update_check_connection_row();
            break;
        case MODE_CHECK_CONNECTION_COL:
            update_check_connection_col();
            break;
        case MODE_CONNECTED:
            update_connected();
            break;
        case MODE_FLAG:
            update_flag();
            break;
        case MODE_ALIGN_TO_SEEK:
            update_align_to_seek();
            break;
        case MODE_SEEK_ROOT:
            update_seek_root();
            break;
        case MODE_FLAG_LIGHT:
            update_flag_light();
            break;
    }
}

int main(void) {

    // init (mandatory)
    pogobot_init();
    srand(pogobot_helper_getRandSeed()); // initialize the random number generator
    //pogobot_infrared_set_power(INFRARED_POWER); // set the power level used to send all the next messages


    
    if (DEBUG_LEVEL) {
        printf("\n");
        printf("[INFO] =-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
        printf("[INFO] =-=- POGOBOT::METADATA -=-=\n");
        printf("[INFO] =-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
        printf("[INFO] INFRARED_POWER         %d\n",INFRARED_POWER);
        printf("[INFO] FQCY                   %d\n",FQCY);
        printf("[INFO] MAX_NB_OF_MSG          %d\n",MAX_NB_OF_MSG);
        printf("[INFO] SEND_MODE_ALLDIRECTION %d\n",SEND_MODE_ALLDIRECTION);
        printf("[INFO] MSG_MODE_FULL_HEADER   %d\n",MSG_MODE_FULL_HEADER);
        printf("[INFO] DEBUG_LEVEL            %d\n",DEBUG_LEVEL);
        printf("[INFO] =-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
    }


    // ********************************************************************************
    // * Initialization: hanabi experimental parameters and initial conditions
    // ********************************************************************************
    time_reference_t mystopwatch; // timer for step synchronization 
    uint32_t microseconds = 0; // counter for step synchronization

    
    // ********************************************************************************
    // * Start-up phase for simultaneous start of the robots when the lights turn off
    // ********************************************************************************

    pogobot_led_setColor(red.r, red.g, red.b); // set initial led color to red

    if (DEBUG_LEVEL == 1) 
        //printf("[I'm Pogobot %d] Photosensor msg_values 0: %d, 1: %d, 2: %d\n", my_pogobot_id, photo0, photo1, photo2);

    pogobot_led_setColor(white.r, white.g, white.b); // set boot led color to white
    sleep(FLAG_BOOT_TIME);
    // pogobot_stopwatch_reset(&timeout_age_watch); // reset of the timer, for age timeout
    
    app_init();

    // ********************************************************************************
    // * Main loop
    // ********************************************************************************

     


    if (DEBUG_LEVEL) {printf("\n------------------------------ initialization ------------------------------\n"); print_all_chemicals_infos(my_pogobot_id, age);}
    while (1)
    {
        pogobot_stopwatch_reset(&mystopwatch); // reset of the timer, for step synchronization

        


        // ********************************************************************************
        // * Reinitialize the age counter in case of stagnation
        // ********************************************************************************

        // seconds = pogobot_stopwatch_get_elapsed_microseconds(&timeout_age_watch) / 1000000; // 1 s = 1000000 microseconds
        // if (seconds >= timeout_age_s) {
        //     if (DEBUG_LEVEL == 1)
        //         printf("[I'm Pogobot %d] [RESET] The age hasn't changed since %lu seconds. Reinitialization of HANABI: age = 0\n", my_pogobot_id, seconds);

        //     age = 0;
        //     led1_status = 0;
        //     pogobot_led_setColors(0, 0, 0, 1); // led 1 off
        //     pogobot_led_setColor(white.r, white.g, white.b); // set upper led color to white
        //     sleep(timeout_age_s);
        //     pogobot_stopwatch_reset(&timeout_age_watch); // reset of the timer, for age timeout
        // }
        
        update_mode();
        // ********************************************************************************
        // * Step synchronize: wait for next step (if not timed out already)
        // ********************************************************************************

        microseconds = pogobot_stopwatch_get_elapsed_microseconds(&mystopwatch);

        if (microseconds < 1000000 / FQCY) { // 1 s = 1000000 microseconds
            if (DEBUG_LEVEL == 2) {
                printf("[TIME] Step took %lu usec. Sleep until next tick.\n", microseconds);
                pogobot_led_setColors(0, 25, 0, 4);
            }
            msleep(((1000000 / FQCY) - microseconds) / 1000); // wait for next step. NB: msleep in ms
        }
        else {
            if (DEBUG_LEVEL == 2) {
                printf("[TIME] Step took %lu usec, should be less than %u usec. [ ### TIME OVERFLOW ### ]\n", microseconds, (1000000 / FQCY));
                pogobot_led_setColors(25, 25, 25, 4); // too slow. Continue directly to next step
            }
        }
    }
}
