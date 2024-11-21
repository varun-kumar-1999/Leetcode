#include<bits/stdc++.h>
using namespace std;
void bag_of_tokens(vector<int>& ar,int pow){
    sort(ar.begin(),ar.end());
    int cur_score = 0,max_score = 0;
    int l = 0,h = ar.size()-1;
    while (l <= h){
        if(pow >= ar[l]){
            pow -= ar[l++];
            cur_score++; 
            max_score = max(max_score,cur_score);  
        }
        else if(cur_score > 0){
            
                cur_score--;
                pow += ar[h--];
                    
        }
        else
            break;
    }
    cout<<max_score;
    
}
int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    vector<int> ar = {200,100};//{100};//{100,200,300,400};
    bag_of_tokens(ar,150);
return 0;
}
