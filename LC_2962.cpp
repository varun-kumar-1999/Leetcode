#include<bits/stdc++.h>
using namespace std;
int solve(vector<int>& ar,int k){
   
    int gmax = *max_element(ar.begin(),ar.end());
    int l = 0,r = 0,ans = 0,mxcnt = 0;
    for(;r<ar.size();r++){
        if(ar[r] == gmax) mxcnt++;
        while(mxcnt >= k){
             if(ar[l++] == gmax)
                mxcnt--;
            ans += ar.size()-r;
        }
           
        
    }

    return ans;
}
int main(){
ios_base::sync_with_stdio(false);
cin.tie(0);
cout.tie(0);
vector<int> ar = {1,3,2,3,3}; 
cout<<solve(ar,2);

return 0;
}